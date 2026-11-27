import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { getSupabaseClient, isSupabaseConfigured } from '../lib/supabase';
import { configureAuthClient } from '../api/client';

const AuthContext = createContext(null);

const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

function isSessionExpired(session) {
  if (!session || typeof session.expires_at !== 'number') {
    return false;
  }
  // Treat tokens within 5 seconds of expiry as stale
  return session.expires_at * 1000 <= Date.now() + 5000;
}

function normalizeAuthError(error, fallbackMessage) {
  if (!error) return fallbackMessage;
  const msg = String(error.message || '').trim();
  if (!msg) return fallbackMessage;
  if (msg.toLowerCase().includes('invalid login credentials')) {
    return 'Invalid email or password. Please check your credentials and try again.';
  }
  if (msg.toLowerCase().includes('email not confirmed')) {
    return 'Please confirm your email address before signing in.';
  }
  return msg;
}

export function AuthProvider({ children, supabaseClient = null }) {
  const resolvedClient = useMemo(
    () => supabaseClient || getSupabaseClient(),
    [supabaseClient]
  );

  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(true);
  const [authError, setAuthError] = useState(null);

  const sessionRef = useRef(null);
  const refreshPromiseRef = useRef(null);

  const applySession = useCallback((nextSession) => {
    sessionRef.current = nextSession || null;
    setSession(nextSession || null);
  }, []);

  const refreshSession = useCallback(async () => {
    if (!resolvedClient) {
      return null;
    }
    if (refreshPromiseRef.current) {
      return await refreshPromiseRef.current;
    }

    refreshPromiseRef.current = (async () => {
      try {
        const { data, error } = await resolvedClient.auth.refreshSession();
        if (error || !data?.session) {
          applySession(null);
          return null;
        }
        applySession(data.session);
        setAuthError(null);
        return data.session;
      } catch {
        applySession(null);
        return null;
      } finally {
        refreshPromiseRef.current = null;
      }
    })();

    return await refreshPromiseRef.current;
  }, [resolvedClient, applySession]);

  const signOut = useCallback(async () => {
    setAuthError(null);
    applySession(null);
    if (!resolvedClient) {
      return;
    }
    try {
      await resolvedClient.auth.signOut();
    } catch {
      // Local session state is already cleared cleanly even if remote revoke fails
    }
  }, [resolvedClient, applySession]);

  const handleUnauthorizedSession = useCallback(async () => {
    applySession(null);
    setAuthError('Your session expired or is no longer valid. Please sign in again.');
    if (resolvedClient) {
      try {
        await resolvedClient.auth.signOut({ scope: 'local' });
      } catch {
        // Ignore signOut cleanup errors on expired local state
      }
    }
  }, [resolvedClient, applySession]);

  // Keep centralized API client synchronized with live Supabase session & refresh
  useEffect(() => {
    const getAccessToken = async () => {
      if (!resolvedClient) {
        return sessionRef.current?.access_token || null;
      }
      try {
        const { data, error } = await resolvedClient.auth.getSession();
        if (error) {
          return null;
        }
        const liveSession = data?.session || null;
        if (liveSession && isSessionExpired(liveSession)) {
          const refreshed = await refreshSession();
          return refreshed?.access_token || null;
        }
        if (liveSession?.access_token !== sessionRef.current?.access_token) {
          applySession(liveSession);
        }
        return liveSession?.access_token || null;
      } catch {
        return null;
      }
    };

    configureAuthClient({
      getAccessToken,
      refreshAccessToken: async () => {
        const refreshed = await refreshSession();
        return refreshed?.access_token || null;
      },
      onUnauthorized: handleUnauthorizedSession,
    });

    return () => {
      configureAuthClient(null);
    };
  }, [resolvedClient, refreshSession, applySession, handleUnauthorizedSession]);

  // Restore Supabase session on startup and subscribe to auth state transitions
  useEffect(() => {
    let active = true;

    if (!resolvedClient) {
      setLoading(false);
      return () => {
        active = false;
      };
    }

    const restoreSession = async () => {
      setLoading(true);
      try {
        const { data, error } = await resolvedClient.auth.getSession();
        if (!active) return;

        if (error) {
          applySession(null);
          setAuthError(normalizeAuthError(error, 'Failed to restore authentication session.'));
          return;
        }

        const initialSession = data?.session || null;
        if (initialSession && isSessionExpired(initialSession)) {
          const refreshed = await refreshSession();
          if (!active) return;
          if (!refreshed) {
            await resolvedClient.auth.signOut({ scope: 'local' }).catch(() => {});
            applySession(null);
            setAuthError('Your previous session expired. Please sign in again.');
          }
          return;
        }

        applySession(initialSession);
      } catch {
        if (active) {
          applySession(null);
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    };

    restoreSession();

    const { data: subscriptionWrap } = resolvedClient.auth.onAuthStateChange((event, nextSession) => {
      if (!active) return;
      if (event === 'SIGNED_OUT' || !nextSession) {
        applySession(null);
        return;
      }
      if (event === 'TOKEN_REFRESHED' || event === 'SIGNED_IN' || event === 'USER_UPDATED' || event === 'INITIAL_SESSION') {
        if (isSessionExpired(nextSession)) {
          refreshSession();
        } else {
          applySession(nextSession);
          setAuthError(null);
        }
      }
    });

    return () => {
      active = false;
      subscriptionWrap?.subscription?.unsubscribe?.();
    };
  }, [resolvedClient, applySession, refreshSession]);

  const signUp = useCallback(
    async ({ email, password }) => {
      setAuthError(null);
      const cleanEmail = String(email || '').trim();
      const rawPassword = String(password || '');

      if (!EMAIL_REGEX.test(cleanEmail)) {
        const err = new Error('Please enter a valid email address.');
        setAuthError(err.message);
        throw err;
      }
      if (rawPassword.length < 6) {
        const err = new Error('Password must be at least 6 characters long.');
        setAuthError(err.message);
        throw err;
      }
      if (!resolvedClient) {
        const err = new Error('Supabase Auth is not configured. Check VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY.');
        setAuthError(err.message);
        throw err;
      }

      const { data, error } = await resolvedClient.auth.signUp({
        email: cleanEmail,
        password: rawPassword,
      });

      if (error) {
        const friendly = normalizeAuthError(error, 'Account registration failed.');
        setAuthError(friendly);
        throw new Error(friendly);
      }

      // Only authenticate if Supabase returned a real session (email confirmation may defer login)
      if (data?.session) {
        applySession(data.session);
      }

      return {
        user: data?.user || null,
        session: data?.session || null,
        requiresConfirmation: Boolean(data?.user && !data?.session),
      };
    },
    [resolvedClient, applySession]
  );

  const signIn = useCallback(
    async ({ email, password }) => {
      setAuthError(null);
      const cleanEmail = String(email || '').trim();
      const rawPassword = String(password || '');

      if (!EMAIL_REGEX.test(cleanEmail)) {
        const err = new Error('Please enter a valid email address.');
        setAuthError(err.message);
        throw err;
      }
      if (!rawPassword) {
        const err = new Error('Password is required.');
        setAuthError(err.message);
        throw err;
      }
      if (!resolvedClient) {
        const err = new Error('Supabase Auth is not configured. Check VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY.');
        setAuthError(err.message);
        throw err;
      }

      const { data, error } = await resolvedClient.auth.signInWithPassword({
        email: cleanEmail,
        password: rawPassword,
      });

      if (error) {
        const friendly = normalizeAuthError(error, 'Sign-in failed.');
        setAuthError(friendly);
        throw new Error(friendly);
      }

      applySession(data?.session || null);
      return {
        user: data?.user || data?.session?.user || null,
        session: data?.session || null,
      };
    },
    [resolvedClient, applySession]
  );

  const value = useMemo(
    () => ({
      user: session?.user || null,
      session,
      accessToken: session?.access_token || null,
      loading,
      authError,
      clearAuthError: () => setAuthError(null),
      isConfigured: Boolean(resolvedClient || isSupabaseConfigured()),
      signUp,
      signIn,
      signOut,
      refreshSession,
    }),
    [session, loading, authError, resolvedClient, signUp, signIn, signOut, refreshSession]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error('useAuth must be used within an AuthProvider.');
  }
  return ctx;
}
