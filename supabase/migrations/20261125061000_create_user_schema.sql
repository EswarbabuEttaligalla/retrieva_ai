-- Base user-owned records for the future authenticated application.

create table public.profiles (
    owner_id uuid primary key references auth.users (id) on delete cascade,
    full_name text not null default '',
    created_at timestamptz not null default pg_catalog.now(),
    updated_at timestamptz not null default pg_catalog.now(),
    constraint profiles_full_name_length check (char_length(full_name) <= 120)
);

create table public.documents (
    id uuid primary key default gen_random_uuid(),
    owner_id uuid not null references auth.users (id) on delete cascade,
    storage_path text not null unique,
    filename text not null,
    file_extension text not null,
    mime_type text not null,
    content_hash text not null,
    file_size_bytes bigint not null,
    processing_status text not null default 'pending',
    embedding_model text,
    total_pages integer not null default 0,
    total_chunks integer not null default 0,
    created_at timestamptz not null default pg_catalog.now(),
    updated_at timestamptz not null default pg_catalog.now(),
    constraint documents_filename_not_empty check (char_length(pg_catalog.btrim(filename)) between 1 and 255),
    constraint documents_extension_supported check (file_extension in ('.pdf', '.txt')),
    constraint documents_filename_extension_matches check (
        pg_catalog.lower(pg_catalog.right(filename, pg_catalog.char_length(file_extension)))
            = pg_catalog.lower(file_extension)
    ),
    constraint documents_mime_matches_extension check (
        (file_extension = '.pdf' and mime_type = 'application/pdf')
        or (file_extension = '.txt' and mime_type = 'text/plain')
    ),
    constraint documents_hash_sha256 check (content_hash ~ '^[0-9a-f]{64}$'),
    constraint documents_size_limit check (file_size_bytes between 1 and 10485760),
    constraint documents_status_supported check (
        processing_status in ('pending', 'uploading', 'indexing', 'ready', 'failed', 'deleting')
    ),
    constraint documents_page_count_nonnegative check (total_pages >= 0),
    constraint documents_chunk_count_nonnegative check (total_chunks >= 0),
    constraint documents_storage_path_scoped check (
        pg_catalog.split_part(storage_path, '/', 1) = owner_id::text
    ),
    constraint documents_owner_hash_unique unique (owner_id, content_hash),
    constraint documents_owner_id_unique unique (owner_id, id)
);

create table public.conversations (
    id uuid primary key default gen_random_uuid(),
    owner_id uuid not null references auth.users (id) on delete cascade,
    title text not null default 'New Conversation',
    created_at timestamptz not null default pg_catalog.now(),
    updated_at timestamptz not null default pg_catalog.now(),
    constraint conversations_title_not_empty check (char_length(pg_catalog.btrim(title)) between 1 and 200),
    constraint conversations_owner_id_unique unique (id, owner_id)
);

create table public.messages (
    id uuid primary key default gen_random_uuid(),
    conversation_id uuid not null,
    owner_id uuid not null references auth.users (id) on delete cascade,
    role text not null,
    content text not null,
    raw_query text,
    rewritten_query text,
    metadata jsonb not null default '{}'::jsonb,
    citations jsonb not null default '[]'::jsonb,
    created_at timestamptz not null default pg_catalog.now(),
    constraint messages_role_supported check (role in ('user', 'assistant')),
    constraint messages_content_not_empty check (char_length(content) > 0),
    constraint messages_citations_array check (pg_catalog.jsonb_typeof(citations) = 'array'),
    constraint messages_conversation_owner_fk
        foreign key (conversation_id, owner_id)
        references public.conversations (id, owner_id)
        on delete cascade
);

create index documents_owner_created_idx
    on public.documents (owner_id, created_at desc);
create index documents_owner_status_idx
    on public.documents (owner_id, processing_status);
create index conversations_owner_updated_idx
    on public.conversations (owner_id, updated_at desc);
create index messages_owner_conversation_created_idx
    on public.messages (owner_id, conversation_id, created_at);

create function public.touch_updated_at()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
    new.updated_at := pg_catalog.now();
    return new;
end;
$$;

revoke all on function public.touch_updated_at() from public, anon, authenticated;

create trigger profiles_touch_updated_at
before update on public.profiles
for each row execute function public.touch_updated_at();

create trigger documents_touch_updated_at
before update on public.documents
for each row execute function public.touch_updated_at();

create trigger conversations_touch_updated_at
before update on public.conversations
for each row execute function public.touch_updated_at();

create function public.create_profile_for_new_auth_user()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
declare
    profile_name text;
begin
    profile_name := pg_catalog.left(
        pg_catalog.btrim(coalesce(new.raw_user_meta_data ->> 'full_name', '')),
        120
    );

    insert into public.profiles (owner_id, full_name)
    values (new.id, profile_name)
    on conflict (owner_id) do nothing;

    return new;
end;
$$;

revoke all on function public.create_profile_for_new_auth_user() from public, anon, authenticated;

create trigger on_retrieva_auth_user_created
after insert on auth.users
for each row execute function public.create_profile_for_new_auth_user();

-- Backfill profiles for accounts that existed before this migration.
insert into public.profiles (owner_id, full_name)
select
    users.id,
    pg_catalog.left(
        pg_catalog.btrim(coalesce(users.raw_user_meta_data ->> 'full_name', '')),
        120
    )
from auth.users as users
on conflict (owner_id) do nothing;

alter table public.profiles enable row level security;
alter table public.documents enable row level security;
alter table public.conversations enable row level security;
alter table public.messages enable row level security;

-- Remove Supabase's default API grants first, then grant only required operations.
revoke all on table public.profiles from public, anon, authenticated;
revoke all on table public.documents from public, anon, authenticated;
revoke all on table public.conversations from public, anon, authenticated;
revoke all on table public.messages from public, anon, authenticated;

grant select on table public.profiles to authenticated;
grant update (full_name) on table public.profiles to authenticated;

grant select on table public.documents to authenticated;

grant select, insert, delete on table public.conversations to authenticated;
grant update (title) on table public.conversations to authenticated;

grant select, insert on table public.messages to authenticated;

create policy profiles_select_own
on public.profiles
for select
to authenticated
using (owner_id = (select auth.uid()));

create policy profiles_update_own
on public.profiles
for update
to authenticated
using (owner_id = (select auth.uid()))
with check (owner_id = (select auth.uid()));

create policy documents_select_own
on public.documents
for select
to authenticated
using (owner_id = (select auth.uid()));

create policy conversations_select_own
on public.conversations
for select
to authenticated
using (owner_id = (select auth.uid()));

create policy conversations_insert_own
on public.conversations
for insert
to authenticated
with check (owner_id = (select auth.uid()));

create policy conversations_update_own
on public.conversations
for update
to authenticated
using (owner_id = (select auth.uid()))
with check (owner_id = (select auth.uid()));

create policy conversations_delete_own
on public.conversations
for delete
to authenticated
using (owner_id = (select auth.uid()));

create policy messages_select_through_owned_conversation
on public.messages
for select
to authenticated
using (
    owner_id = (select auth.uid())
    and exists (
        select 1
        from public.conversations
        where conversations.id = messages.conversation_id
          and conversations.owner_id = (select auth.uid())
    )
);

-- Authenticated clients may add user turns only. Assistant turns must be written
-- by a future backend path that verifies ownership; no direct client update/delete
-- permission is granted on message rows.
create policy messages_insert_user_turn_in_owned_conversation
on public.messages
for insert
to authenticated
with check (
    owner_id = (select auth.uid())
    and role = 'user'
    and exists (
        select 1
        from public.conversations
        where conversations.id = messages.conversation_id
          and conversations.owner_id = (select auth.uid())
    )
);
