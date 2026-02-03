-- Migration: Add strategy_id to chat table
-- This creates a 1:N relationship between strategies and chats
-- Each chat can optionally be associated with a strategy

-- Add strategy_id column to chat table
ALTER TABLE chat 
ADD COLUMN IF NOT EXISTS strategy_id INTEGER REFERENCES strategies(id) ON DELETE CASCADE;

-- Create index for better query performance
CREATE INDEX IF NOT EXISTS idx_chat_strategy_id ON chat(strategy_id);

-- Extend chat_type column to support longer enum values (strategy, generic)
ALTER TABLE chat ALTER COLUMN chat_type TYPE VARCHAR(20);

-- Comment for documentation
COMMENT ON COLUMN chat.strategy_id IS 'FK to strategies table - allows each strategy to have multiple associated chats';
