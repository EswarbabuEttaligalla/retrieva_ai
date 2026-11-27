-- Cross-user pgTAP security tests for owner-scoped RPCs & backend-only HMAC authorization.
begin;

select plan(35);

-- Provision a test-only Vault secret inside the rolled-back transaction.
select vault.create_secret('pgtap-test-backend-rpc-hmac-secret', 'retrieva_backend_rpc_hmac_secret');

insert into auth.users (
    id, aud, role, email, encrypted_password, email_confirmed_at,
    raw_app_meta_data, raw_user_meta_data, created_at, updated_at
)
values
    (
        '10000000-0000-4000-8000-000000000001', 'authenticated', 'authenticated',
        'rpc-user-a@example.test', '', pg_catalog.now(),
        '{"provider":"email","providers":["email"]}'::jsonb, '{"full_name":"RPC User A"}'::jsonb,
        pg_catalog.now(), pg_catalog.now()
    ),
    (
        '10000000-0000-4000-8000-000000000002', 'authenticated', 'authenticated',
        'rpc-user-b@example.test', '', pg_catalog.now(),
        '{"provider":"email","providers":["email"]}'::jsonb, '{"full_name":"RPC User B"}'::jsonb,
        pg_catalog.now(), pg_catalog.now()
    )
on conflict (id) do nothing;

insert into public.conversations (id, owner_id, title)
values
    ('30000000-0000-4000-8000-000000000011', '10000000-0000-4000-8000-000000000001', 'RPC User A Chat')
on conflict (id) do nothing;

-- 1. Anon cannot execute ANY of the 4 owner-scoped RPCs
set local role anon;
select set_config('request.jwt.claims', '{}', true);

select throws_ok(
    $$select * from public.create_user_document(
        '20000000-0000-4000-8000-000000000011'::uuid,
        '10000000-0000-4000-8000-000000000001/20000000-0000-4000-8000-000000000011/test.pdf',
        'test.pdf', '.pdf', 'application/pdf', pg_catalog.repeat('e', 64), 128, 'pending', null
    )$$,
    '42501',
    null,
    'anon cannot execute create_user_document'
);
select throws_ok(
    $$select * from public.update_user_document_status('20000000-0000-4000-8000-000000000011', 'ready')$$,
    '42501',
    null,
    'anon cannot execute update_user_document_status'
);
select throws_ok(
    $$select * from public.delete_user_document('20000000-0000-4000-8000-000000000011')$$,
    '42501',
    null,
    'anon cannot execute delete_user_document'
);
select throws_ok(
    $$select * from public.record_assistant_message('30000000-0000-4000-8000-000000000011', 'spoof')$$,
    '42501',
    null,
    'anon cannot execute record_assistant_message'
);

-- 2. Authenticated role WITHOUT auth.uid() claim must fail closed with 42501
set local role authenticated;
select set_config('request.jwt.claims', '{}', true);

select throws_ok(
    $$select * from public.create_user_document('20000000-0000-4000-8000-000000000011'::uuid, 'a/b/c.pdf', 'c.pdf', '.pdf', 'application/pdf', pg_catalog.repeat('a', 64), 10, 'pending', null)$$,
    '42501',
    null,
    'authenticated missing sub claim fails closed in create_user_document'
);
select throws_ok(
    $$select * from public.record_assistant_message('30000000-0000-4000-8000-000000000011', 'no-sub')$$,
    '42501',
    null,
    'authenticated missing sub claim fails closed in record_assistant_message'
);

-- 3. Authenticated User A WITHOUT backend proof is denied (direct browser RPC blocked)
select set_config(
    'request.jwt.claims',
    '{"sub":"10000000-0000-4000-8000-000000000001","role":"authenticated","aud":"authenticated"}',
    true
);

select throws_ok(
    $$select * from public.create_user_document(
        '20000000-0000-4000-8000-000000000011'::uuid,
        '10000000-0000-4000-8000-000000000001/20000000-0000-4000-8000-000000000011/test.pdf',
        'test.pdf', '.pdf', 'application/pdf', pg_catalog.repeat('e', 64), 128, 'uploading', null
    )$$,
    '42501',
    null,
    'direct authenticated create_user_document without backend proof is denied'
);

select throws_ok(
    $$select * from public.update_user_document_status('20000000-0000-4000-8000-000000000011'::uuid, 'ready')$$,
    '42501',
    null,
    'direct authenticated update_user_document_status without backend proof is denied'
);

select throws_ok(
    $$select * from public.delete_user_document('20000000-0000-4000-8000-000000000011'::uuid)$$,
    '42501',
    null,
    'direct authenticated delete_user_document without backend proof is denied'
);

select throws_ok(
    $$select * from public.record_assistant_message('30000000-0000-4000-8000-000000000011'::uuid, 'Direct browser spoof')$$,
    '42501',
    null,
    'direct authenticated record_assistant_message without backend proof is denied'
);

select throws_ok(
    $$select * from private.rpc_proof_nonces$$,
    '42501',
    null,
    'authenticated cannot access private.rpc_proof_nonces'
);

-- 4. Invalid, expired, cross-user, and cross-operation proofs are denied
select throws_ok(
    $$select * from public.create_user_document(
        '20000000-0000-4000-8000-000000000011'::uuid,
        '10000000-0000-4000-8000-000000000001/20000000-0000-4000-8000-000000000011/test.pdf',
        'test.pdf', '.pdf', 'application/pdf', pg_catalog.repeat('e', 64), 128, 'uploading', null,
        pg_catalog.repeat('0', 64),
        'pgtap-nonce-invalid-0001',
        pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60
    )$$,
    '42501',
    null,
    'invalid backend proof is denied'
);

select throws_ok(
    $$select * from public.create_user_document(
        '20000000-0000-4000-8000-000000000011'::uuid,
        '10000000-0000-4000-8000-000000000001/20000000-0000-4000-8000-000000000011/test.pdf',
        'test.pdf', '.pdf', 'application/pdf', pg_catalog.repeat('e', 64), 128, 'uploading', null,
        pg_catalog.encode(
            extensions.hmac(
                pg_catalog.convert_to(
                    '10000000-0000-4000-8000-000000000001|create_user_document|20000000-0000-4000-8000-000000000011|'
                    || pg_catalog.encode(extensions.digest(pg_catalog.convert_to(
                        '82:10000000-0000-4000-8000-000000000001/20000000-0000-4000-8000-000000000011/test.pdf|8:test.pdf|4:.pdf|15:application/pdf|64:' || pg_catalog.repeat('e', 64) || '|3:128|9:uploading|0:', 'UTF8'
                    ), 'sha256'), 'hex')
                    || '|' || (pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint - 10)::text
                    || '|pgtap-nonce-expired-0001',
                    'UTF8'
                ),
                pg_catalog.convert_to('pgtap-test-backend-rpc-hmac-secret', 'UTF8'),
                'sha256'
            ),
            'hex'
        ),
        'pgtap-nonce-expired-0001',
        pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint - 10
    )$$,
    '42501',
    null,
    'expired backend proof is denied'
);

select throws_ok(
    $$select * from public.update_user_document_status(
        '20000000-0000-4000-8000-000000000011'::uuid,
        'ready',
        null, null, null, null,
        pg_catalog.encode(
            extensions.hmac(
                pg_catalog.convert_to(
                    '10000000-0000-4000-8000-000000000001|create_user_document|20000000-0000-4000-8000-000000000011|'
                    || pg_catalog.encode(extensions.digest(pg_catalog.convert_to(
                        '5:ready|0:|0:|0:|0:', 'UTF8'
                    ), 'sha256'), 'hex')
                    || '|' || (pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60)::text
                    || '|pgtap-nonce-crossop-0001',
                    'UTF8'
                ),
                pg_catalog.convert_to('pgtap-test-backend-rpc-hmac-secret', 'UTF8'),
                'sha256'
            ),
            'hex'
        ),
        'pgtap-nonce-crossop-0001',
        pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60
    )$$,
    '42501',
    null,
    'proof minted for create_user_document cannot authorize update_user_document_status'
);

-- 5. User A backend-signed create & update succeed and replay is denied
select is(
    (select owner_id from public.create_user_document(
        '20000000-0000-4000-8000-000000000011'::uuid,
        '10000000-0000-4000-8000-000000000001/20000000-0000-4000-8000-000000000011/test.pdf',
        'test.pdf',
        '.pdf',
        'application/pdf',
        pg_catalog.repeat('e', 64),
        128,
        'uploading',
        null,
        pg_catalog.encode(
            extensions.hmac(
                pg_catalog.convert_to(
                    '10000000-0000-4000-8000-000000000001|create_user_document|20000000-0000-4000-8000-000000000011|'
                    || pg_catalog.encode(extensions.digest(pg_catalog.convert_to(
                        '82:10000000-0000-4000-8000-000000000001/20000000-0000-4000-8000-000000000011/test.pdf|8:test.pdf|4:.pdf|15:application/pdf|64:' || pg_catalog.repeat('e', 64) || '|3:128|9:uploading|0:', 'UTF8'
                    ), 'sha256'), 'hex')
                    || '|' || (pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60)::text
                    || '|pgtap-nonce-create-0001',
                    'UTF8'
                ),
                pg_catalog.convert_to('pgtap-test-backend-rpc-hmac-secret', 'UTF8'),
                'sha256'
            ),
            'hex'
        ),
        'pgtap-nonce-create-0001',
        pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60
    )),
    '10000000-0000-4000-8000-000000000001'::uuid,
    'User A backend-signed create_user_document succeeds scoped to auth.uid()'
);

select throws_ok(
    $$select * from public.create_user_document(
        '20000000-0000-4000-8000-000000000011'::uuid,
        '10000000-0000-4000-8000-000000000001/20000000-0000-4000-8000-000000000011/test.pdf',
        'test.pdf', '.pdf', 'application/pdf', pg_catalog.repeat('e', 64), 128, 'uploading', null,
        pg_catalog.encode(
            extensions.hmac(
                pg_catalog.convert_to(
                    '10000000-0000-4000-8000-000000000001|create_user_document|20000000-0000-4000-8000-000000000011|'
                    || pg_catalog.encode(extensions.digest(pg_catalog.convert_to(
                        '82:10000000-0000-4000-8000-000000000001/20000000-0000-4000-8000-000000000011/test.pdf|8:test.pdf|4:.pdf|15:application/pdf|64:' || pg_catalog.repeat('e', 64) || '|3:128|9:uploading|0:', 'UTF8'
                    ), 'sha256'), 'hex')
                    || '|' || (pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60)::text
                    || '|pgtap-nonce-create-0001',
                    'UTF8'
                ),
                pg_catalog.convert_to('pgtap-test-backend-rpc-hmac-secret', 'UTF8'),
                'sha256'
            ),
            'hex'
        ),
        'pgtap-nonce-create-0001',
        pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60
    )$$,
    '42501',
    null,
    'replayed backend proof nonce is denied'
);

select is(
    (select processing_status from public.update_user_document_status(
        '20000000-0000-4000-8000-000000000011'::uuid,
        'ready',
        2,
        4,
        'sentence-transformers/all-MiniLM-L6-v2',
        null,
        pg_catalog.encode(
            extensions.hmac(
                pg_catalog.convert_to(
                    '10000000-0000-4000-8000-000000000001|update_user_document_status|20000000-0000-4000-8000-000000000011|'
                    || pg_catalog.encode(extensions.digest(pg_catalog.convert_to(
                        '5:ready|1:2|1:4|38:sentence-transformers/all-MiniLM-L6-v2|0:', 'UTF8'
                    ), 'sha256'), 'hex')
                    || '|' || (pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60)::text
                    || '|pgtap-nonce-update-0001',
                    'UTF8'
                ),
                pg_catalog.convert_to('pgtap-test-backend-rpc-hmac-secret', 'UTF8'),
                'sha256'
            ),
            'hex'
        ),
        'pgtap-nonce-update-0001',
        pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60
    )),
    'ready',
    'User A backend-signed update_user_document_status succeeds'
);

select throws_ok(
    $$select * from public.record_assistant_message(
        '30000000-0000-4000-8000-000000000011'::uuid,
        'Tampered assistant response [Source 1].',
        '{"retrieval_mode":"hybrid"}'::jsonb,
        '[{"document_id":"20000000-0000-4000-8000-000000000011","filename":"test.pdf","page_number":1}]'::jsonb,
        pg_catalog.encode(
            extensions.hmac(
                pg_catalog.convert_to(
                    '10000000-0000-4000-8000-000000000001|record_assistant_message|30000000-0000-4000-8000-000000000011|'
                    || pg_catalog.encode(extensions.digest(pg_catalog.convert_to(
                        '39:Grounded assistant response [Source 1].'
                        || '|32:o:{14:retrieval_mode=s:6:hybrid}'
                        || '|107:a:[o:{11:document_id=s:36:20000000-0000-4000-8000-000000000011,8:filename=s:8:test.pdf,11:page_number=d:1}]',
                        'UTF8'
                    ), 'sha256'), 'hex')
                    || '|' || (pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60)::text
                    || '|pgtap-nonce-msg-0001',
                    'UTF8'
                ),
                pg_catalog.convert_to('pgtap-test-backend-rpc-hmac-secret', 'UTF8'),
                'sha256'
            ),
            'hex'
        ),
        'pgtap-nonce-msg-0001',
        pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60
    )$$,
    '42501',
    null,
    'tampered content invalidates record_assistant_message proof'
);

select is(
    (select role from public.record_assistant_message(
        '30000000-0000-4000-8000-000000000011'::uuid,
        'Grounded assistant response [Source 1].',
        '{"retrieval_mode":"hybrid"}'::jsonb,
        '[{"document_id":"20000000-0000-4000-8000-000000000011","filename":"test.pdf","page_number":1}]'::jsonb,
        pg_catalog.encode(
            extensions.hmac(
                pg_catalog.convert_to(
                    '10000000-0000-4000-8000-000000000001|record_assistant_message|30000000-0000-4000-8000-000000000011|'
                    || pg_catalog.encode(extensions.digest(pg_catalog.convert_to(
                        '39:Grounded assistant response [Source 1].'
                        || '|32:o:{14:retrieval_mode=s:6:hybrid}'
                        || '|107:a:[o:{11:document_id=s:36:20000000-0000-4000-8000-000000000011,8:filename=s:8:test.pdf,11:page_number=d:1}]',
                        'UTF8'
                    ), 'sha256'), 'hex')
                    || '|' || (pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60)::text
                    || '|pgtap-nonce-msg-0001',
                    'UTF8'
                ),
                pg_catalog.convert_to('pgtap-test-backend-rpc-hmac-secret', 'UTF8'),
                'sha256'
            ),
            'hex'
        ),
        'pgtap-nonce-msg-0001',
        pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60
    )),
    'assistant',
    'User A backend call can record assistant message for own conversation'
);

select throws_ok(
    $$select * from public.record_assistant_message(
        '30000000-0000-4000-8000-000000000011'::uuid,
        'Grounded assistant response [Source 1].',
        '{"retrieval_mode":"hybrid"}'::jsonb,
        '[{"document_id":"20000000-0000-4000-8000-000000000011","filename":"test.pdf","page_number":1}]'::jsonb,
        pg_catalog.encode(
            extensions.hmac(
                pg_catalog.convert_to(
                    '10000000-0000-4000-8000-000000000001|record_assistant_message|30000000-0000-4000-8000-000000000011|'
                    || pg_catalog.encode(extensions.digest(pg_catalog.convert_to(
                        '39:Grounded assistant response [Source 1].'
                        || '|32:o:{14:retrieval_mode=s:6:hybrid}'
                        || '|107:a:[o:{11:document_id=s:36:20000000-0000-4000-8000-000000000011,8:filename=s:8:test.pdf,11:page_number=d:1}]',
                        'UTF8'
                    ), 'sha256'), 'hex')
                    || '|' || (pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60)::text
                    || '|pgtap-nonce-msg-0001',
                    'UTF8'
                ),
                pg_catalog.convert_to('pgtap-test-backend-rpc-hmac-secret', 'UTF8'),
                'sha256'
            ),
            'hex'
        ),
        'pgtap-nonce-msg-0001',
        pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60
    )$$,
    '42501',
    null,
    'replayed record_assistant_message proof nonce is denied'
);

-- Direct table writes remain denied even for authenticated User A
select throws_ok(
    $$insert into public.documents (owner_id, storage_path, filename, file_extension, mime_type, content_hash, file_size_bytes) values ('10000000-0000-4000-8000-000000000001', '10000000-0000-4000-8000-000000000001/x/direct.txt', 'direct.txt', '.txt', 'text/plain', pg_catalog.repeat('1', 64), 10)$$,
    '42501',
    null,
    'direct public.documents insert remains blocked'
);
select throws_ok(
    $$insert into public.messages (conversation_id, owner_id, role, content) values ('30000000-0000-4000-8000-000000000011', '10000000-0000-4000-8000-000000000001', 'assistant', 'direct')$$,
    '42501',
    null,
    'direct assistant message insert remains blocked by RLS'
);

-- 6. Invalid inputs fail closed
select throws_ok(
    $$select * from public.create_user_document(
        '20000000-0000-4000-8000-000000000013'::uuid,
        '10000000-0000-4000-8000-000000000001/20000000-0000-4000-8000-000000000013/../evil.pdf',
        'evil.pdf', '.pdf', 'application/pdf', pg_catalog.repeat('a', 64), 64, 'pending', null,
        pg_catalog.encode(
            extensions.hmac(
                pg_catalog.convert_to(
                    '10000000-0000-4000-8000-000000000001|create_user_document|20000000-0000-4000-8000-000000000013|'
                    || pg_catalog.encode(extensions.digest(pg_catalog.convert_to(
                        '85:10000000-0000-4000-8000-000000000001/20000000-0000-4000-8000-000000000013/../evil.pdf|8:evil.pdf|4:.pdf|15:application/pdf|64:' || pg_catalog.repeat('a', 64) || '|2:64|7:pending|0:', 'UTF8'
                    ), 'sha256'), 'hex')
                    || '|' || (pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60)::text
                    || '|pgtap-nonce-evil-0001',
                    'UTF8'
                ),
                pg_catalog.convert_to('pgtap-test-backend-rpc-hmac-secret', 'UTF8'),
                'sha256'
            ),
            'hex'
        ),
        'pgtap-nonce-evil-0001',
        pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60
    )$$,
    '23514',
    null,
    'traversal segments in p_storage_path are rejected'
);
select throws_ok(
    $$select * from public.update_user_document_status(
        '20000000-0000-4000-8000-000000000011'::uuid,
        'bogus_status',
        null, null, null, null,
        pg_catalog.encode(
            extensions.hmac(
                pg_catalog.convert_to(
                    '10000000-0000-4000-8000-000000000001|update_user_document_status|20000000-0000-4000-8000-000000000011|'
                    || pg_catalog.encode(extensions.digest(pg_catalog.convert_to(
                        '12:bogus_status|0:|0:|0:|0:', 'UTF8'
                    ), 'sha256'), 'hex')
                    || '|' || (pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60)::text
                    || '|pgtap-nonce-bogus-0001',
                    'UTF8'
                ),
                pg_catalog.convert_to('pgtap-test-backend-rpc-hmac-secret', 'UTF8'),
                'sha256'
            ),
            'hex'
        ),
        'pgtap-nonce-bogus-0001',
        pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60
    )$$,
    '23514',
    null,
    'invalid processing_status is rejected'
);
select throws_ok(
    $$select * from public.record_assistant_message('30000000-0000-4000-8000-000000000011'::uuid, '   ')$$,
    '22023',
    null,
    'whitespace-only assistant message content is rejected'
);
select throws_ok(
    $$select * from public.record_assistant_message('30000000-0000-4000-8000-000000000011'::uuid, 'Valid reply', '[]'::jsonb, '[]'::jsonb)$$,
    '22023',
    null,
    'non-object assistant metadata is rejected'
);
select throws_ok(
    $$select * from public.record_assistant_message('30000000-0000-4000-8000-000000000011'::uuid, 'Valid reply', '{}'::jsonb, '{}'::jsonb)$$,
    '22023',
    null,
    'non-array assistant citations are rejected'
);

-- 7. User B cannot mutate or delete User A records, nor use User A's proof
select set_config(
    'request.jwt.claims',
    '{"sub":"10000000-0000-4000-8000-000000000002","role":"authenticated","aud":"authenticated"}',
    true
);

select throws_ok(
    $$select * from public.update_user_document_status(
        '20000000-0000-4000-8000-000000000011'::uuid,
        'failed',
        null, null, null, 'tampered',
        pg_catalog.encode(
            extensions.hmac(
                pg_catalog.convert_to(
                    '10000000-0000-4000-8000-000000000001|update_user_document_status|20000000-0000-4000-8000-000000000011|'
                    || pg_catalog.encode(extensions.digest(pg_catalog.convert_to(
                        '6:failed|0:|0:|0:|8:tampered', 'UTF8'
                    ), 'sha256'), 'hex')
                    || '|' || (pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60)::text
                    || '|pgtap-nonce-crossuser-0001',
                    'UTF8'
                ),
                pg_catalog.convert_to('pgtap-test-backend-rpc-hmac-secret', 'UTF8'),
                'sha256'
            ),
            'hex'
        ),
        'pgtap-nonce-crossuser-0001',
        pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60
    )$$,
    '42501',
    null,
    'User A proof cannot authorize User B'
);

select is(
    (select count(*)::int from public.update_user_document_status(
        '20000000-0000-4000-8000-000000000011'::uuid,
        'failed',
        null, null, null, 'tampered',
        pg_catalog.encode(
            extensions.hmac(
                pg_catalog.convert_to(
                    '10000000-0000-4000-8000-000000000002|update_user_document_status|20000000-0000-4000-8000-000000000011|'
                    || pg_catalog.encode(extensions.digest(pg_catalog.convert_to(
                        '6:failed|0:|0:|0:|8:tampered', 'UTF8'
                    ), 'sha256'), 'hex')
                    || '|' || (pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60)::text
                    || '|pgtap-nonce-userb-0001',
                    'UTF8'
                ),
                pg_catalog.convert_to('pgtap-test-backend-rpc-hmac-secret', 'UTF8'),
                'sha256'
            ),
            'hex'
        ),
        'pgtap-nonce-userb-0001',
        pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60
    )),
    0,
    'Even with User B backend proof, update_user_document_status returns 0 rows for User A document'
);

select is(
    (select count(*)::int from public.delete_user_document(
        '20000000-0000-4000-8000-000000000011'::uuid,
        pg_catalog.encode(
            extensions.hmac(
                pg_catalog.convert_to(
                    '10000000-0000-4000-8000-000000000002|delete_user_document|20000000-0000-4000-8000-000000000011|'
                    || pg_catalog.encode(extensions.digest(pg_catalog.convert_to(
                        '36:20000000-0000-4000-8000-000000000011', 'UTF8'
                    ), 'sha256'), 'hex')
                    || '|' || (pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60)::text
                    || '|pgtap-nonce-delb-0001',
                    'UTF8'
                ),
                pg_catalog.convert_to('pgtap-test-backend-rpc-hmac-secret', 'UTF8'),
                'sha256'
            ),
            'hex'
        ),
        'pgtap-nonce-delb-0001',
        pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60
    )),
    0,
    'Even with User B backend proof, delete_user_document returns 0 rows for User A document'
);

select throws_ok(
    $$select * from public.record_assistant_message(
        '30000000-0000-4000-8000-000000000011'::uuid,
        'cross-tenant',
        '{}'::jsonb,
        '[]'::jsonb,
        pg_catalog.encode(
            extensions.hmac(
                pg_catalog.convert_to(
                    '10000000-0000-4000-8000-000000000001|record_assistant_message|30000000-0000-4000-8000-000000000011|'
                    || pg_catalog.encode(extensions.digest(pg_catalog.convert_to(
                        '12:cross-tenant|4:o:{}|4:a:[]', 'UTF8'
                    ), 'sha256'), 'hex')
                    || '|' || (pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60)::text
                    || '|pgtap-nonce-msg-cross-0001',
                    'UTF8'
                ),
                pg_catalog.convert_to('pgtap-test-backend-rpc-hmac-secret', 'UTF8'),
                'sha256'
            ),
            'hex'
        ),
        'pgtap-nonce-msg-cross-0001',
        pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60
    )$$,
    '42501',
    null,
    'User A proof cannot authorize User B on record_assistant_message'
);

select throws_ok(
    $$select * from public.record_assistant_message(
        '30000000-0000-4000-8000-000000000011'::uuid,
        'cross-tenant',
        '{}'::jsonb,
        '[]'::jsonb,
        pg_catalog.encode(
            extensions.hmac(
                pg_catalog.convert_to(
                    '10000000-0000-4000-8000-000000000002|record_assistant_message|30000000-0000-4000-8000-000000000011|'
                    || pg_catalog.encode(extensions.digest(pg_catalog.convert_to(
                        '12:cross-tenant|4:o:{}|4:a:[]', 'UTF8'
                    ), 'sha256'), 'hex')
                    || '|' || (pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60)::text
                    || '|pgtap-nonce-msg-userb-0001',
                    'UTF8'
                ),
                pg_catalog.convert_to('pgtap-test-backend-rpc-hmac-secret', 'UTF8'),
                'sha256'
            ),
            'hex'
        ),
        'pgtap-nonce-msg-userb-0001',
        pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60
    )$$,
    '42501',
    null,
    'User B cannot record assistant message into User A conversation even with User B proof'
);

-- Storage path spoofing for User A prefix under User B token is blocked by constraint
select throws_ok(
    $$select * from public.create_user_document(
        '20000000-0000-4000-8000-000000000012'::uuid,
        '10000000-0000-4000-8000-000000000001/20000000-0000-4000-8000-000000000012/spoof.pdf',
        'spoof.pdf', '.pdf', 'application/pdf', pg_catalog.repeat('f', 64), 128, 'pending', null,
        pg_catalog.encode(
            extensions.hmac(
                pg_catalog.convert_to(
                    '10000000-0000-4000-8000-000000000002|create_user_document|20000000-0000-4000-8000-000000000012|'
                    || pg_catalog.encode(extensions.digest(pg_catalog.convert_to(
                        '83:10000000-0000-4000-8000-000000000001/20000000-0000-4000-8000-000000000012/spoof.pdf|9:spoof.pdf|4:.pdf|15:application/pdf|64:' || pg_catalog.repeat('f', 64) || '|3:128|7:pending|0:', 'UTF8'
                    ), 'sha256'), 'hex')
                    || '|' || (pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60)::text
                    || '|pgtap-nonce-spoof-0001',
                    'UTF8'
                ),
                pg_catalog.convert_to('pgtap-test-backend-rpc-hmac-secret', 'UTF8'),
                'sha256'
            ),
            'hex'
        ),
        'pgtap-nonce-spoof-0001',
        pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60
    )$$,
    '23514',
    null,
    'User B cannot insert document pointing to User A storage prefix'
);

-- 8. Switch back to User A: document still intact and User A can delete own document
select set_config(
    'request.jwt.claims',
    '{"sub":"10000000-0000-4000-8000-000000000001","role":"authenticated","aud":"authenticated"}',
    true
);

select is(
    (select processing_status from public.documents where id = '20000000-0000-4000-8000-000000000011'::uuid),
    'ready',
    'User A document status remained unchanged after User B attack'
);

select is(
    (select count(*)::int from public.delete_user_document(
        '20000000-0000-4000-8000-000000000011'::uuid,
        pg_catalog.encode(
            extensions.hmac(
                pg_catalog.convert_to(
                    '10000000-0000-4000-8000-000000000001|delete_user_document|20000000-0000-4000-8000-000000000011|'
                    || pg_catalog.encode(extensions.digest(pg_catalog.convert_to(
                        '36:20000000-0000-4000-8000-000000000011', 'UTF8'
                    ), 'sha256'), 'hex')
                    || '|' || (pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60)::text
                    || '|pgtap-nonce-dela-0001',
                    'UTF8'
                ),
                pg_catalog.convert_to('pgtap-test-backend-rpc-hmac-secret', 'UTF8'),
                'sha256'
            ),
            'hex'
        ),
        'pgtap-nonce-dela-0001',
        pg_catalog.floor(EXTRACT(EPOCH FROM pg_catalog.clock_timestamp()))::bigint + 60
    )),
    1,
    'User A can delete own document via backend-signed RPC'
);

select * from finish();
rollback;
