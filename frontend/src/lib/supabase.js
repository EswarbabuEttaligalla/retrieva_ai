import { createClient } from '@supabase/supabase-js';

let _supabaseClient = null;
let _overrideClient = null;

/**
 * Read Vite/Node frontend Supabase configuration from environment variables only.
 * Never hardcodes credentials and never accepts service-role keys.
 */
export function getSupabaseConfig() {
  const metaEnv = typeof import.meta !== 'undefined' && import.meta.env ? import.meta.env : {};
  const procEnv = typeof process !== 'undefined' && process.env ? process.env : {};

  const supabaseUrl = String(metaEnv.VITE_SUPABASE_URL || procEnv.VITE_SUPABASE_URL || '').trim();
  const supabaseAnonKey = String(metaEnv.VITE_SUPABASE_ANON_KEY || procEnv.VITE_SUPABASE_ANON_KEY || '').trim();

  const isPlaceholder =
    !supabaseAnonKey ||
    supabaseAnonKey.includes('your-supabase') ||
    supabaseAnonKey.includes('placeholder');

  return {
    supabaseUrl,
    supabaseAnonKey: isPlaceholder ? '' : supabaseAnonKey,
    isConfigured: Boolean(supabaseUrl && !isPlaceholder),
  };
}

/**
 * Return the singleton Supabase JS client configured for browser Auth,
 * or an injected mock client during automated frontend tests.
 */
export function getSupabaseClient() {
  if (_overrideClient !== null) {
    return _overrideClient;
  }

  if (_supabaseClient !== null) {
    return _supabaseClient;
  }

  const { supabaseUrl, supabaseAnonKey, isConfigured } = getSupabaseConfig();
  if (!isConfigured) {
    return null;
  }

  _supabaseClient = createClient(supabaseUrl, supabaseAnonKey, {
    auth: {
      autoRefreshToken: true,
      persistSession: true,
      detectSessionInUrl: true,
    },
  });

  return _supabaseClient;
}

/**
 * Test-only hook to inject a deterministic Supabase Auth mock or reset singleton state.
 */
export function setSupabaseClientForTesting(mockClient) {
  _overrideClient = mockClient || null;
  if (!mockClient) {
    _supabaseClient = null;
  }
}

export function isSupabaseConfigured() {
  return Boolean(getSupabaseClient() || getSupabaseConfig().isConfigured);
}
