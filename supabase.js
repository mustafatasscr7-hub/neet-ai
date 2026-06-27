import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js/+esm'

const SUPABASE_URL = 'https://hvhnfttrfouajlyhvunq.supabase.co'
const SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imh2aG5mdHRyZm91YWpseWh2dW5xIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzg5NDI1NTIsImV4cCI6MjA5NDUxODU1Mn0.8RaSu5yiTpQCEUGKrFO2y6oiRXSqdBehIz933j9Z7WA'

export const supabase = createClient(SUPABASE_URL, SUPABASE_KEY)