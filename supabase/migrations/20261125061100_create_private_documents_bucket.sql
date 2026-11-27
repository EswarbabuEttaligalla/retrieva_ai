-- Private object storage for user-owned PDF and TXT documents.

insert into storage.buckets (
    id,
    name,
    public,
    file_size_limit,
    allowed_mime_types
)
values (
    'retrieva-user-documents',
    'retrieva-user-documents',
    false,
    10485760,
    array[
        'application/pdf',
        'text/plain'
    ]::text[]
)
-- Preserve pre-existing bucket settings; the validation below rejects conflicts.
on conflict do nothing;

do $verify_documents_bucket$
begin
    if not exists (
        select 1
        from storage.buckets
        where id = 'retrieva-user-documents'
          and name = 'retrieva-user-documents'
          and public is false
          and file_size_limit = 10485760
          and cardinality(allowed_mime_types) = 2
          and allowed_mime_types @> array['application/pdf', 'text/plain']::text[]
    ) then
        raise exception
            'Storage bucket retrieva-user-documents is missing or conflicts with the required private 10 MiB PDF/TXT configuration; reconcile it manually before retrying.';
    end if;
end;
$verify_documents_bucket$;

create policy retrieva_documents_select_own
on storage.objects
for select
to authenticated
using (
    bucket_id = 'retrieva-user-documents'
    and (storage.foldername(name))[1] = (select auth.uid()::text)
);

create policy retrieva_documents_insert_own
on storage.objects
for insert
to authenticated
with check (
    bucket_id = 'retrieva-user-documents'
    and (storage.foldername(name))[1] = (select auth.uid()::text)
);

create policy retrieva_documents_update_own
on storage.objects
for update
to authenticated
using (
    bucket_id = 'retrieva-user-documents'
    and (storage.foldername(name))[1] = (select auth.uid()::text)
)
with check (
    bucket_id = 'retrieva-user-documents'
    and (storage.foldername(name))[1] = (select auth.uid()::text)
);

create policy retrieva_documents_delete_own
on storage.objects
for delete
to authenticated
using (
    bucket_id = 'retrieva-user-documents'
    and (storage.foldername(name))[1] = (select auth.uid()::text)
);
