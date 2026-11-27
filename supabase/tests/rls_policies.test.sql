begin;

select plan(38);

insert into auth.users (
    id,
    aud,
    role,
    email,
    encrypted_password,
    email_confirmed_at,
    raw_app_meta_data,
    raw_user_meta_data,
    created_at,
    updated_at
)
values
    (
        '10000000-0000-4000-8000-000000000001',
        'authenticated',
        'authenticated',
        'rls-user-a@example.test',
        '',
        pg_catalog.now(),
        '{"provider":"email","providers":["email"]}'::jsonb,
        '{"full_name":"RLS User A"}'::jsonb,
        pg_catalog.now(),
        pg_catalog.now()
    ),
    (
        '10000000-0000-4000-8000-000000000002',
        'authenticated',
        'authenticated',
        'rls-user-b@example.test',
        '',
        pg_catalog.now(),
        '{"provider":"email","providers":["email"]}'::jsonb,
        '{"full_name":"RLS User B"}'::jsonb,
        pg_catalog.now(),
        pg_catalog.now()
    )
on conflict (id) do nothing;

insert into public.documents (
    id, owner_id, storage_path, filename, file_extension, mime_type,
    content_hash, file_size_bytes, processing_status
)
values
    (
        '20000000-0000-4000-8000-000000000001',
        '10000000-0000-4000-8000-000000000001',
        '10000000-0000-4000-8000-000000000001/20000000-0000-4000-8000-000000000001/example.txt',
        'user-a.txt', '.txt', 'text/plain', pg_catalog.repeat('a', 64), 10, 'ready'
    ),
    (
        '20000000-0000-4000-8000-000000000002',
        '10000000-0000-4000-8000-000000000002',
        '10000000-0000-4000-8000-000000000002/20000000-0000-4000-8000-000000000002/example.txt',
        'user-b.txt', '.txt', 'text/plain', pg_catalog.repeat('b', 64), 10, 'ready'
    );

insert into public.conversations (id, owner_id, title)
values
    ('30000000-0000-4000-8000-000000000001', '10000000-0000-4000-8000-000000000001', 'User A chat'),
    ('30000000-0000-4000-8000-000000000002', '10000000-0000-4000-8000-000000000002', 'User B chat');

insert into public.messages (id, conversation_id, owner_id, role, content)
values
    (
        '40000000-0000-4000-8000-000000000001',
        '30000000-0000-4000-8000-000000000001',
        '10000000-0000-4000-8000-000000000001',
        'user',
        'Question from user A'
    ),
    (
        '40000000-0000-4000-8000-000000000002',
        '30000000-0000-4000-8000-000000000002',
        '10000000-0000-4000-8000-000000000002',
        'user',
        'Question from user B'
    );

select lives_ok(
    $$insert into public.documents (
        owner_id, storage_path, filename, file_extension, mime_type,
        content_hash, file_size_bytes
      ) values (
        '10000000-0000-4000-8000-000000000001',
        '10000000-0000-4000-8000-000000000001/20000000-0000-4000-8000-000000000005/case-insensitive.TXT',
        'case-insensitive.TXT', '.txt', 'text/plain', pg_catalog.repeat('d', 64), 10
      )$$,
    'filename extension matching is case-insensitive'
);
delete from public.documents
where owner_id = '10000000-0000-4000-8000-000000000001'
  and filename = 'case-insensitive.TXT';
select throws_ok(
    $$insert into public.documents (
        owner_id, storage_path, filename, file_extension, mime_type,
        content_hash, file_size_bytes
      ) values (
        '10000000-0000-4000-8000-000000000001',
        '10000000-0000-4000-8000-000000000001/20000000-0000-4000-8000-000000000006/wrong-extension.pdf',
        'wrong-extension.pdf', '.txt', 'text/plain', pg_catalog.repeat('e', 64), 10
      )$$,
    '23514',
    null,
    'filename extension must match file_extension'
);

select throws_ok(
    $$insert into public.documents (
        owner_id, storage_path, filename, file_extension, mime_type,
        content_hash, file_size_bytes
      ) values (
        '10000000-0000-4000-8000-000000000001',
        '10000000-0000-4000-8000-000000000001/20000000-0000-4000-8000-000000000003/invalid.txt',
        'invalid.txt', '.txt', 'application/pdf', pg_catalog.repeat('c', 64), 10
      )$$,
    '23514',
    null,
    'document MIME type must match its extension'
);
select throws_ok(
    $$insert into public.messages (conversation_id, owner_id, role, content, citations)
      values (
        '30000000-0000-4000-8000-000000000001',
        '10000000-0000-4000-8000-000000000001',
        'user',
        'Invalid citations shape',
        '{}'::jsonb
      )$$,
    '23514',
    null,
    'citations must be a JSON array'
);

select is(
    (select public from storage.buckets where id = 'retrieva-user-documents'),
    false,
    'document bucket is private'
);
select ok(
    exists (
        select 1
        from storage.buckets
        where id = 'retrieva-user-documents'
          and name = 'retrieva-user-documents'
          and public is false
          and file_size_limit = 10485760
          and cardinality(allowed_mime_types) = 2
          and allowed_mime_types @> array['application/pdf', 'text/plain']::text[]
    ),
    'document bucket has the exact private 10 MiB PDF/TXT configuration'
);

set local role anon;
select throws_ok(
    $$select * from public.documents$$,
    '42501',
    null,
    'anonymous users cannot read documents'
);
select throws_ok(
    $$select * from public.profiles$$,
    '42501',
    null,
    'anonymous users cannot read profiles'
);
select throws_ok(
    $$select * from public.conversations$$,
    '42501',
    null,
    'anonymous users cannot read conversations'
);
select throws_ok(
    $$select * from public.messages$$,
    '42501',
    null,
    'anonymous users cannot read messages'
);
reset role;

select pg_catalog.set_config(
    'request.jwt.claim.sub',
    '10000000-0000-4000-8000-000000000001',
    true
);
select pg_catalog.set_config(
    'request.jwt.claims',
    '{"sub":"10000000-0000-4000-8000-000000000001","role":"authenticated","aud":"authenticated"}',
    true
);
set local role authenticated;

select is(auth.uid(), '10000000-0000-4000-8000-000000000001'::uuid, 'user A identity comes from auth.uid()');
select is((select count(*)::integer from public.profiles), 1, 'user A sees only their profile');
select is((select count(*)::integer from public.documents), 1, 'user A sees only their document metadata');
select is((select count(*)::integer from public.conversations), 1, 'user A sees only their conversations');
select is((select count(*)::integer from public.messages), 1, 'user A sees only messages in their conversation');
select is(
    (
        select count(*)::integer
        from public.conversations
        where id = '30000000-0000-4000-8000-000000000002'
    ),
    0,
    'user A cannot select user B exact conversation'
);
select is(
    (
        select count(*)::integer
        from public.documents
        where id = '20000000-0000-4000-8000-000000000002'
    ),
    0,
    'user A cannot select user B exact document metadata'
);
select is(
    (
        select count(*)::integer
        from public.messages
        where id = '40000000-0000-4000-8000-000000000002'
    ),
    0,
    'user A cannot select user B exact message'
);

select lives_ok(
    $$insert into public.conversations (id, owner_id, title)
      values ('30000000-0000-4000-8000-000000000003',
              '10000000-0000-4000-8000-000000000001',
              'User A new chat')$$,
    'user A can create a conversation owned by user A'
);
select throws_ok(
    $$insert into public.conversations (owner_id, title)
      values ('10000000-0000-4000-8000-000000000002', 'Forged owner')$$,
    '42501',
    null,
    'user A cannot create a conversation owned by user B'
);
select lives_ok(
    $$update public.conversations
      set title = 'Renamed by A'
      where id = '30000000-0000-4000-8000-000000000001'$$,
    'user A can rename their own conversation'
);
select throws_ok(
    $$update public.conversations
      set owner_id = '10000000-0000-4000-8000-000000000002'
      where id = '30000000-0000-4000-8000-000000000001'$$,
    '42501',
    null,
    'user A cannot update the conversation owner_id column'
);
select is(
    (
        with updated as (
            update public.conversations
            set title = 'Unauthorized rename'
            where id = '30000000-0000-4000-8000-000000000002'
            returning id
        )
        select count(*)::integer from updated
    ),
    0,
    'user A cannot update user B conversation'
);
select throws_ok(
    $$insert into public.messages (conversation_id, owner_id, role, content)
      values ('30000000-0000-4000-8000-000000000002',
              '10000000-0000-4000-8000-000000000001',
              'user',
              'Cross-owner message')$$,
    '42501',
    null,
    'user A cannot attach a message to user B conversation'
);
select lives_ok(
    $$insert into public.messages (conversation_id, owner_id, role, content)
      values ('30000000-0000-4000-8000-000000000001',
              '10000000-0000-4000-8000-000000000001',
              'user',
              'Another question from A')$$,
    'user A can add a user message to their own conversation'
);
select throws_ok(
    $$insert into public.messages (conversation_id, owner_id, role, content)
      values ('30000000-0000-4000-8000-000000000001',
              '10000000-0000-4000-8000-000000000001',
              'assistant',
              'Forged assistant answer')$$,
    '42501',
    null,
    'authenticated clients cannot insert assistant messages directly'
);
select throws_ok(
    $$insert into public.documents (
        owner_id, storage_path, filename, file_extension, mime_type,
        content_hash, file_size_bytes
      ) values (
        '10000000-0000-4000-8000-000000000001',
        '10000000-0000-4000-8000-000000000001/new/file.txt',
        'file.txt', '.txt', 'text/plain', pg_catalog.repeat('c', 64), 10
      )$$,
    '42501',
    null,
    'authenticated clients cannot write document metadata directly'
);
select lives_ok(
    $$update public.profiles set full_name = 'Updated User A'$$,
    'user A can update their own profile fields'
);
select throws_ok(
    $$update public.profiles
      set owner_id = '10000000-0000-4000-8000-000000000002'$$,
    '42501',
    null,
    'user A cannot update profile ownership'
);

reset role;
select pg_catalog.set_config(
    'request.jwt.claim.sub',
    '10000000-0000-4000-8000-000000000002',
    true
);
select pg_catalog.set_config(
    'request.jwt.claims',
    '{"sub":"10000000-0000-4000-8000-000000000002","role":"authenticated","aud":"authenticated"}',
    true
);
set local role authenticated;

select is(auth.uid(), '10000000-0000-4000-8000-000000000002'::uuid, 'user B identity comes from auth.uid()');
select is((select count(*)::integer from public.profiles), 1, 'user B sees only their profile');
select is((select count(*)::integer from public.documents), 1, 'user B sees only their document metadata');
select is((select count(*)::integer from public.conversations), 1, 'user B sees only their conversation');
select is((select count(*)::integer from public.messages), 1, 'user B sees only messages in their conversation');
select is(
    (
        select count(*)::integer
        from public.conversations
        where id = '30000000-0000-4000-8000-000000000001'
    ),
    0,
    'user B cannot select user A exact conversation'
);
select is(
    (
        select count(*)::integer
        from public.documents
        where id = '20000000-0000-4000-8000-000000000001'
    ),
    0,
    'user B cannot select user A exact document metadata'
);
select is(
    (
        select count(*)::integer
        from public.messages
        where id = '40000000-0000-4000-8000-000000000001'
    ),
    0,
    'user B cannot select user A exact message'
);
select is(
    (select title from public.conversations where id = '30000000-0000-4000-8000-000000000002'),
    'User B chat',
    'user A actions did not modify user B conversation'
);

select * from finish();
rollback;
