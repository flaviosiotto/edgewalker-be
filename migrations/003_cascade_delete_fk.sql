-- Migration 003: Add CASCADE DELETE to foreign keys
-- This ensures that when a data source is deleted, all related records are also deleted

-- Drop existing foreign key constraints and recreate with CASCADE

-- 1. symbol_cache.source_id -> data_sources.id
ALTER TABLE symbol_cache 
DROP CONSTRAINT IF EXISTS symbol_cache_source_id_fkey;

ALTER TABLE symbol_cache
ADD CONSTRAINT symbol_cache_source_id_fkey 
FOREIGN KEY (source_id) REFERENCES data_sources(id) ON DELETE CASCADE;

-- 2. symbol_sync_log.source_id -> data_sources.id
ALTER TABLE symbol_sync_log 
DROP CONSTRAINT IF EXISTS symbol_sync_log_source_id_fkey;

ALTER TABLE symbol_sync_log
ADD CONSTRAINT symbol_sync_log_source_id_fkey 
FOREIGN KEY (source_id) REFERENCES data_sources(id) ON DELETE CASCADE;

-- Verify constraints
SELECT 
    tc.constraint_name,
    tc.table_name,
    kcu.column_name,
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name,
    rc.delete_rule
FROM information_schema.table_constraints AS tc
JOIN information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
JOIN information_schema.constraint_column_usage AS ccu
    ON ccu.constraint_name = tc.constraint_name
JOIN information_schema.referential_constraints AS rc
    ON rc.constraint_name = tc.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY'
AND tc.table_name IN ('symbol_cache', 'symbol_sync_log');
