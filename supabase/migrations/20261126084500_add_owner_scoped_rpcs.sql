-- Milestone 3.2 / 3.9 (Local Proposal — NOT APPLIED REMOTELY)
-- Purpose:
--   1. Add nullable error_message column to public.documents for failed ingestion diagnostics.
--   2. Provide narrowly scoped SECURITY DEFINER RPCs so authenticated backend requests
--      using the caller's own JWT (auth.uid()) can manage document lifecycle rows and
--      record assistant messages without granting broad table INSERT/UPDATE/DELETE
--      privileges to browser clients or requiring a service-role key.
--   3. Require a short-lived, single-use, payload-bound HMAC-SHA256 backend authorization
--      proof (verified against Supabase Vault) on all 4 owner-scoped RPCs
--      (create_user_document, update_user_document_status, delete_user_document,
--      record_assistant_message) so direct browser PostgREST RPC calls fail closed.
--
-- RLS & Grant Implications:
--   - Existing RLS policies and table grants on public.documents and public.messages
--     remain unchanged and strictly restrictive:
--       * authenticated still has SELECT-only on public.documents.
--       * authenticated still has SELECT + INSERT(role='user') on public.messages.
--   - Every RPC explicitly checks auth.uid() IS NOT NULL and scopes all reads/writes
--     to owner_id = auth.uid().
--   - All functions use `set search_path = ''` and schema-qualified references.
--   - private schema, nonce table, and proof-verification helpers have zero grants to
--     public, anon, or authenticated.
--   - EXECUTE on public RPCs is revoked from PUBLIC and anon, and granted strictly to authenticated.
--
-- Prerequisites in Supabase Vault before live invocation:
--   select vault.create_secret('<high-entropy-secret>', 'retrieva_backend_rpc_hmac_secret');
--
-- Rollback Considerations:
--   drop function if exists public.record_assistant_message(uuid, text, jsonb, jsonb, text, text, bigint);
--   drop function if exists public.delete_user_document(uuid, text, text, bigint);
--   drop function if exists public.update_user_document_status(uuid, text, integer, integer, text, text, text, text, bigint);
--   drop function if exists public.create_user_document(uuid, text, text, text, text, text, bigint, text, text, text, text, bigint);
--   drop function if exists private.verify_backend_rpc_proof(uuid, text, uuid, text, text, text, bigint);
--   drop function if exists private.canonical_jsonb(jsonb);
--   drop function if exists private.lp_field(text);
--   drop table if exists private.rpc_proof_nonces;
--   drop schema if exists private;
--   alter table public.documents drop constraint if exists documents_error_message_length;
--   alter table public.documents drop column if exists error_message;

create extension if not exists pgcrypto with schema extensions;

create schema if not exists private;
revoke all on schema private from public, anon, authenticated;

create table if not exists private.rpc_proof_nonces (
    nonce text primary key,
    expires_at bigint not null,
    created_at timestamptz not null default pg_catalog.now()
);

revoke all on table private.rpc_proof_nonces from public, anon, authenticated;

create function private.lp_field(p_val text)
returns text
language sql
immutable
set search_path = ''
as $$
    select coalesce(pg_catalog.char_length(p_val), 0)::text || ':' || coalesce(p_val, '');
$$;

revoke all on function private.lp_field(text) from public, anon, authenticated;

create function private.canonical_jsonb(p_val jsonb)
returns text
language plpgsql
immutable
set search_path = ''
as $$
declare
    v_type text;
    v_num text;
    v_items text;
begin
    if p_val is null then
        return 'n';
    end if;

    v_type := pg_catalog.jsonb_typeof(p_val);
    if v_type = 'null' then
        return 'n';
    elsif v_type = 'boolean' then
        return 'b:' || (p_val #>> '{}');
    elsif v_type = 'number' then
        v_num := p_val #>> '{}';
        if pg_catalog.strpos(v_num, '.') > 0 then
            v_num := pg_catalog.regexp_replace(v_num, '0+$', '');
            v_num := pg_catalog.regexp_replace(v_num, '\.$', '');
        end if;
        if v_num = '-0' then
            v_num := '0';
        end if;
        return 'd:' || v_num;
    elsif v_type = 'string' then
        return 's:' || private.lp_field(p_val #>> '{}');
    elsif v_type = 'array' then
        select coalesce(pg_catalog.string_agg(private.canonical_jsonb(elem.val), ',' order by elem.ord), '')
        into v_items
        from pg_catalog.jsonb_array_elements(p_val) with ordinality as elem(val, ord);
        return 'a:[' || v_items || ']';
    elsif v_type = 'object' then
        select coalesce(
            pg_catalog.string_agg(
                private.lp_field(kv.key) || '=' || private.canonical_jsonb(kv.val),
                ',' order by kv.key collate "C"
            ),
            ''
        )
        into v_items
        from pg_catalog.jsonb_each(p_val) as kv(key, val);
        return 'o:{' || v_items || '}';
    end if;

    raise exception 'Unsupported JSONB type: %', v_type using errcode = '22023';
end;
$$;

revoke all on function private.canonical_jsonb(jsonb) from public, anon, authenticated;

create function private.verify_backend_rpc_proof(
    p_uid uuid,
    p_operation text,
    p_document_id uuid,
    p_payload_digest text,
    p_backend_proof text,
    p_proof_nonce text,
    p_proof_expires_at bigint
)
returns void
language plpgsql
security definer
set search_path = ''
as $$
declare
    v_now_epoch bigint;
    v_secret text;
    v_message text;
    v_expected text;
    v_provided text;
    v_diff integer := 0;
    v_idx integer;
begin
    if p_uid is null or p_document_id is null or p_operation is null then
        raise exception 'Authentication and document context required' using errcode = '42501';
    end if;

    if p_backend_proof is null
       or p_proof_nonce is null
       or p_proof_expires_at is null
       or p_proof_nonce <> pg_catalog.btrim(p_proof_nonce)
       or pg_catalog.char_length(p_proof_nonce) not between 16 and 128 then
        raise exception 'Backend authorization proof required' using errcode = '42501';
    end if;

    v_provided := pg_catalog.lower(pg_catalog.btrim(p_backend_proof));
    if v_provided !~ '^[0-9a-f]{64}$' then
        raise exception 'Malformed backend authorization proof' using errcode = '42501';
    end if;

    v_now_epoch := pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint;
    if p_proof_expires_at < v_now_epoch then
        raise exception 'Backend authorization proof expired' using errcode = '42501';
    end if;
    if p_proof_expires_at > v_now_epoch + 300 then
        raise exception 'Backend authorization proof TTL exceeds maximum window' using errcode = '42501';
    end if;

    select ds.decrypted_secret
    into v_secret
    from vault.decrypted_secrets as ds
    where ds.name = 'retrieva_backend_rpc_hmac_secret'
    limit 1;

    if v_secret is null or pg_catalog.char_length(pg_catalog.btrim(v_secret)) = 0 then
        raise exception 'Backend RPC authorization secret is not configured' using errcode = '42501';
    end if;

    v_message := p_uid::text
        || '|' || p_operation
        || '|' || p_document_id::text
        || '|' || p_payload_digest
        || '|' || p_proof_expires_at::text
        || '|' || p_proof_nonce;

    v_expected := pg_catalog.encode(
        extensions.hmac(
            pg_catalog.convert_to(v_message, 'UTF8'),
            pg_catalog.convert_to(v_secret, 'UTF8'),
            'sha256'
        ),
        'hex'
    );

    -- Constant-time comparison over fixed 64-char hex digests.
    for v_idx in 1..64 loop
        v_diff := v_diff | (
            pg_catalog.ascii(pg_catalog.substr(v_expected, v_idx, 1))
            # pg_catalog.ascii(pg_catalog.substr(v_provided, v_idx, 1))
        );
    end loop;

    if v_diff <> 0 then
        raise exception 'Invalid backend authorization proof' using errcode = '42501';
    end if;

    delete from private.rpc_proof_nonces
    where expires_at < v_now_epoch;

    insert into private.rpc_proof_nonces (nonce, expires_at)
    values (p_proof_nonce, p_proof_expires_at)
    on conflict (nonce) do nothing;

    if not found then
        raise exception 'Backend authorization proof has already been used' using errcode = '42501';
    end if;
end;
$$;

revoke all on function private.verify_backend_rpc_proof(uuid, text, uuid, text, text, text, bigint)
    from public, anon, authenticated;

alter table public.documents
    add column if not exists error_message text;

do $add_error_message_constraint$
begin
    if not exists (
        select 1
        from pg_constraint
        where conname = 'documents_error_message_length'
          and conrelid = 'public.documents'::regclass
    ) then
        alter table public.documents
            add constraint documents_error_message_length
            check (error_message is null or pg_catalog.char_length(error_message) <= 500);
    end if;
end;
$add_error_message_constraint$;

create function public.create_user_document(
    p_document_id uuid,
    p_storage_path text,
    p_filename text,
    p_file_extension text,
    p_mime_type text,
    p_content_hash text,
    p_file_size_bytes bigint,
    p_processing_status text default 'pending',
    p_embedding_model text default null,
    p_backend_proof text default null,
    p_proof_nonce text default null,
    p_proof_expires_at bigint default null
)
returns public.documents
language plpgsql
security definer
set search_path = ''
as $$
declare
    v_uid uuid;
    v_payload_digest text;
    v_row public.documents;
begin
    v_uid := auth.uid();
    if v_uid is null then
        raise exception 'Authentication required' using errcode = '42501';
    end if;

    if p_document_id is null then
        raise exception 'Document ID is required' using errcode = '22023';
    end if;

    v_payload_digest := pg_catalog.encode(
        extensions.digest(
            pg_catalog.convert_to(
                private.lp_field(p_storage_path)
                || '|' || private.lp_field(p_filename)
                || '|' || private.lp_field(pg_catalog.lower(p_file_extension))
                || '|' || private.lp_field(pg_catalog.lower(p_mime_type))
                || '|' || private.lp_field(pg_catalog.lower(p_content_hash))
                || '|' || private.lp_field(p_file_size_bytes::text)
                || '|' || private.lp_field(p_processing_status)
                || '|' || private.lp_field(p_embedding_model),
                'UTF8'
            ),
            'sha256'
        ),
        'hex'
    );

    perform private.verify_backend_rpc_proof(
        v_uid,
        'create_user_document',
        p_document_id,
        v_payload_digest,
        p_backend_proof,
        p_proof_nonce,
        p_proof_expires_at
    );

    if p_storage_path is null
       or pg_catalog.strpos(p_storage_path, '..') > 0
       or pg_catalog.split_part(p_storage_path, '/', 1) <> v_uid::text
       or pg_catalog.split_part(p_storage_path, '/', 2) <> p_document_id::text
       or pg_catalog.char_length(pg_catalog.btrim(pg_catalog.split_part(p_storage_path, '/', 3))) = 0 then
        raise exception 'Invalid owner-scoped storage path' using errcode = '23514';
    end if;

    insert into public.documents (
        id,
        owner_id,
        storage_path,
        filename,
        file_extension,
        mime_type,
        content_hash,
        file_size_bytes,
        processing_status,
        embedding_model
    )
    values (
        p_document_id,
        v_uid,
        p_storage_path,
        p_filename,
        pg_catalog.lower(p_file_extension),
        pg_catalog.lower(p_mime_type),
        pg_catalog.lower(p_content_hash),
        p_file_size_bytes,
        p_processing_status,
        p_embedding_model
    )
    returning * into v_row;

    return v_row;
end;
$$;

create function public.update_user_document_status(
    p_document_id uuid,
    p_processing_status text,
    p_total_pages integer default null,
    p_total_chunks integer default null,
    p_embedding_model text default null,
    p_error_message text default null,
    p_backend_proof text default null,
    p_proof_nonce text default null,
    p_proof_expires_at bigint default null
)
returns setof public.documents
language plpgsql
security definer
set search_path = ''
as $$
declare
    v_uid uuid;
    v_payload_digest text;
begin
    v_uid := auth.uid();
    if v_uid is null then
        raise exception 'Authentication required' using errcode = '42501';
    end if;

    if p_document_id is null then
        raise exception 'Document ID is required' using errcode = '22023';
    end if;

    v_payload_digest := pg_catalog.encode(
        extensions.digest(
            pg_catalog.convert_to(
                private.lp_field(p_processing_status)
                || '|' || private.lp_field(p_total_pages::text)
                || '|' || private.lp_field(p_total_chunks::text)
                || '|' || private.lp_field(p_embedding_model)
                || '|' || private.lp_field(p_error_message),
                'UTF8'
            ),
            'sha256'
        ),
        'hex'
    );

    perform private.verify_backend_rpc_proof(
        v_uid,
        'update_user_document_status',
        p_document_id,
        v_payload_digest,
        p_backend_proof,
        p_proof_nonce,
        p_proof_expires_at
    );

    return query
    update public.documents as d
    set
        processing_status = p_processing_status,
        total_pages = coalesce(p_total_pages, d.total_pages),
        total_chunks = coalesce(p_total_chunks, d.total_chunks),
        embedding_model = coalesce(p_embedding_model, d.embedding_model),
        error_message = case
            when p_processing_status = 'ready' then null
            else coalesce(p_error_message, d.error_message)
        end
    where d.id = p_document_id
      and d.owner_id = v_uid
    returning d.*;
end;
$$;

create function public.delete_user_document(
    p_document_id uuid,
    p_backend_proof text default null,
    p_proof_nonce text default null,
    p_proof_expires_at bigint default null
)
returns setof public.documents
language plpgsql
security definer
set search_path = ''
as $$
declare
    v_uid uuid;
    v_payload_digest text;
begin
    v_uid := auth.uid();
    if v_uid is null then
        raise exception 'Authentication required' using errcode = '42501';
    end if;

    if p_document_id is null then
        raise exception 'Document ID is required' using errcode = '22023';
    end if;

    v_payload_digest := pg_catalog.encode(
        extensions.digest(
            pg_catalog.convert_to(private.lp_field(p_document_id::text), 'UTF8'),
            'sha256'
        ),
        'hex'
    );

    perform private.verify_backend_rpc_proof(
        v_uid,
        'delete_user_document',
        p_document_id,
        v_payload_digest,
        p_backend_proof,
        p_proof_nonce,
        p_proof_expires_at
    );

    return query
    delete from public.documents as d
    where d.id = p_document_id
      and d.owner_id = v_uid
    returning d.*;
end;
$$;

create function public.record_assistant_message(
    p_conversation_id uuid,
    p_content text,
    p_metadata jsonb default '{}'::jsonb,
    p_citations jsonb default '[]'::jsonb,
    p_backend_proof text default null,
    p_proof_nonce text default null,
    p_proof_expires_at bigint default null
)
returns public.messages
language plpgsql
security definer
set search_path = ''
as $$
declare
    v_uid uuid;
    v_metadata jsonb;
    v_citations jsonb;
    v_payload_digest text;
    v_msg public.messages;
begin
    v_uid := auth.uid();
    if v_uid is null then
        raise exception 'Authentication required' using errcode = '42501';
    end if;

    if p_conversation_id is null then
        raise exception 'Conversation ID is required' using errcode = '22023';
    end if;

    if p_content is null or pg_catalog.char_length(pg_catalog.btrim(p_content)) = 0 then
        raise exception 'Assistant message content cannot be empty' using errcode = '22023';
    end if;

    if p_metadata is not null and pg_catalog.jsonb_typeof(p_metadata) <> 'object' then
        raise exception 'Message metadata must be a JSON object' using errcode = '22023';
    end if;

    if p_citations is not null and pg_catalog.jsonb_typeof(p_citations) <> 'array' then
        raise exception 'Message citations must be a JSON array' using errcode = '22023';
    end if;

    v_metadata := coalesce(p_metadata, '{}'::jsonb);
    v_citations := coalesce(p_citations, '[]'::jsonb);

    v_payload_digest := pg_catalog.encode(
        extensions.digest(
            pg_catalog.convert_to(
                private.lp_field(p_content)
                || '|' || private.lp_field(private.canonical_jsonb(v_metadata))
                || '|' || private.lp_field(private.canonical_jsonb(v_citations)),
                'UTF8'
            ),
            'sha256'
        ),
        'hex'
    );

    perform private.verify_backend_rpc_proof(
        v_uid,
        'record_assistant_message',
        p_conversation_id,
        v_payload_digest,
        p_backend_proof,
        p_proof_nonce,
        p_proof_expires_at
    );

    if not exists (
        select 1
        from public.conversations as c
        where c.id = p_conversation_id
          and c.owner_id = v_uid
    ) then
        raise exception 'Conversation not found or not owned by caller' using errcode = '42501';
    end if;

    insert into public.messages (
        conversation_id,
        owner_id,
        role,
        content,
        metadata,
        citations
    )
    values (
        p_conversation_id,
        v_uid,
        'assistant',
        p_content,
        v_metadata,
        v_citations
    )
    returning * into v_msg;

    update public.conversations
    set updated_at = pg_catalog.now()
    where id = p_conversation_id
      and owner_id = v_uid;

    return v_msg;
end;
$$;

revoke all on function public.create_user_document(uuid, text, text, text, text, text, bigint, text, text, text, text, bigint)
    from public, anon, authenticated;
revoke all on function public.update_user_document_status(uuid, text, integer, integer, text, text, text, text, bigint)
    from public, anon, authenticated;
revoke all on function public.delete_user_document(uuid, text, text, bigint)
    from public, anon, authenticated;
revoke all on function public.record_assistant_message(uuid, text, jsonb, jsonb, text, text, bigint)
    from public, anon, authenticated;

grant execute on function public.create_user_document(uuid, text, text, text, text, text, bigint, text, text, text, text, bigint)
    to authenticated;
grant execute on function public.update_user_document_status(uuid, text, integer, integer, text, text, text, text, bigint)
    to authenticated;
grant execute on function public.delete_user_document(uuid, text, text, bigint)
    to authenticated;
grant execute on function public.record_assistant_message(uuid, text, jsonb, jsonb, text, text, bigint)
    to authenticated;
