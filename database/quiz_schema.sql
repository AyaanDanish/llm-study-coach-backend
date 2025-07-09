-- Database schema for quiz functionality
-- Run these commands in your Supabase SQL editor

-- Create quizzes table
CREATE TABLE IF NOT EXISTS quizzes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    subject TEXT NOT NULL,
    questions JSONB NOT NULL,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    content_hash TEXT NOT NULL, -- Reference to study_notes via content_hash instead of study_material_id
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create quiz_attempts table
CREATE TABLE IF NOT EXISTS quiz_attempts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    quiz_id UUID NOT NULL REFERENCES quizzes(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    score DECIMAL(5,2) NOT NULL CHECK (score >= 0 AND score <= 100),
    total_questions INTEGER NOT NULL,
    answers JSONB NOT NULL,
    completed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_quizzes_user_id ON quizzes(user_id);
CREATE INDEX IF NOT EXISTS idx_quizzes_content_hash ON quizzes(content_hash);
CREATE INDEX IF NOT EXISTS idx_quizzes_created_at ON quizzes(created_at);
CREATE INDEX IF NOT EXISTS idx_quiz_attempts_quiz_id ON quiz_attempts(quiz_id);
CREATE INDEX IF NOT EXISTS idx_quiz_attempts_user_id ON quiz_attempts(user_id);
CREATE INDEX IF NOT EXISTS idx_quiz_attempts_completed_at ON quiz_attempts(completed_at);

-- Enable Row Level Security (RLS)
ALTER TABLE quizzes ENABLE ROW LEVEL SECURITY;
ALTER TABLE quiz_attempts ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for quizzes
CREATE POLICY "Users can view their own quizzes" ON quizzes
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own quizzes" ON quizzes
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own quizzes" ON quizzes
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own quizzes" ON quizzes
    FOR DELETE USING (auth.uid() = user_id);

-- Create RLS policies for quiz_attempts
CREATE POLICY "Users can view their own quiz attempts" ON quiz_attempts
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own quiz attempts" ON quiz_attempts
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own quiz attempts" ON quiz_attempts
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own quiz attempts" ON quiz_attempts
    FOR DELETE USING (auth.uid() = user_id);

-- Add trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_quizzes_updated_at 
    BEFORE UPDATE ON quizzes 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Migration: Update existing quizzes table to use content_hash instead of study_material_id
-- Only run this if you have an existing quizzes table with study_material_id

-- Step 1: Add content_hash column
-- ALTER TABLE quizzes ADD COLUMN IF NOT EXISTS content_hash TEXT;

-- Step 2: Populate content_hash from study_materials table
-- UPDATE quizzes 
-- SET content_hash = (
--     SELECT content_hash 
--     FROM study_materials 
--     WHERE study_materials.id = quizzes.study_material_id
-- ) 
-- WHERE content_hash IS NULL;

-- Step 3: Make content_hash NOT NULL after populating
-- ALTER TABLE quizzes ALTER COLUMN content_hash SET NOT NULL;

-- Step 4: Drop old column and index
-- DROP INDEX IF EXISTS idx_quizzes_study_material_id;
-- ALTER TABLE quizzes DROP COLUMN IF EXISTS study_material_id;

-- Step 5: Create new index
-- CREATE INDEX IF NOT EXISTS idx_quizzes_content_hash ON quizzes(content_hash);

-- Sample data structure for questions JSONB field:
-- [
--   {
--     "id": "q_1",
--     "question": "What is the capital of France?",
--     "options": ["London", "Paris", "Berlin", "Madrid"],
--     "correct_answer": 1,
--     "explanation": "Paris is the capital and largest city of France."
--   }
-- ]

-- Sample data structure for answers JSONB field:
-- {
--   "q_1": 1,
--   "q_2": 0,
--   "q_3": 2,
--   "q_4": 1,
--   "q_5": 3
-- }
