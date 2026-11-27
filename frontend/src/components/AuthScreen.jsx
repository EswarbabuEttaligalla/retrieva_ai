import React, { useState } from 'react';
import { AlertCircle, CheckCircle2, Loader2, Lock, Mail, Sparkles } from 'lucide-react';

export function AuthLoadingScreen() {
  return (
    <div
      className="min-h-screen w-screen bg-slate-950 text-slate-100 flex flex-col items-center justify-center p-6"
      role="status"
      aria-live="polite"
    >
      <div className="w-12 h-12 rounded-2xl bg-gradient-to-tr from-brand-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-brand-500/20 mb-4">
        <Sparkles className="w-6 h-6 text-white" />
      </div>
      <div className="flex items-center space-x-2 text-sm text-slate-300 font-medium">
        <Loader2 className="w-4 h-4 animate-spin text-brand-400" />
        <span>Restoring your secure Retrieva AI session...</span>
      </div>
    </div>
  );
}

export default function AuthScreen({
  onSignIn,
  onSignUp,
  authError = null,
  onClearError = () => {},
  isConfigured = true,
}) {
  const [mode, setMode] = useState('login'); // 'login' | 'signup'
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [localError, setLocalError] = useState(null);
  const [confirmationMessage, setConfirmationMessage] = useState(null);

  const activeError = localError || authError;

  const switchMode = (nextMode) => {
    setMode(nextMode);
    setLocalError(null);
    setConfirmationMessage(null);
    onClearError();
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLocalError(null);
    setConfirmationMessage(null);
    onClearError();

    const cleanEmail = email.trim();
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(cleanEmail)) {
      setLocalError('Please enter a valid email address.');
      return;
    }

    if (mode === 'signup') {
      if (password.length < 6) {
        setLocalError('Password must be at least 6 characters long.');
        return;
      }
      if (password !== confirmPassword) {
        setLocalError('Passwords do not match.');
        return;
      }
    } else if (!password) {
      setLocalError('Password is required.');
      return;
    }

    setSubmitting(true);
    try {
      if (mode === 'signup') {
        const res = await onSignUp({ email: cleanEmail, password });
        if (res?.requiresConfirmation) {
          setConfirmationMessage(
            `Confirmation link sent to ${cleanEmail}. Verify your email before signing in.`
          );
          setPassword('');
          setConfirmPassword('');
          setMode('login');
        }
      } else {
        await onSignIn({ email: cleanEmail, password });
      }
    } catch (err) {
      setLocalError(err.message || 'Authentication failed.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen w-screen bg-slate-950 text-slate-100 flex items-center justify-center p-4 sm:p-6">
      <div className="w-full max-w-md bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl overflow-hidden">
        {/* Brand Header */}
        <div className="p-6 border-b border-slate-800 bg-slate-900/70 flex items-center space-x-3">
          <div className="w-11 h-11 rounded-xl bg-gradient-to-tr from-brand-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-brand-500/20 shrink-0">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white tracking-tight">Retrieva AI</h1>
            <p className="text-xs text-slate-400">
              Private multi-tenant document intelligence &amp; grounded QA
            </p>
          </div>
        </div>

        {/* Mode Tabs */}
        <div className="px-6 pt-5">
          <div
            role="tablist"
            aria-label="Authentication Mode"
            className="grid grid-cols-2 gap-1 p-1 bg-slate-950 border border-slate-800 rounded-xl text-xs font-medium"
          >
            <button
              type="button"
              role="tab"
              aria-selected={mode === 'login'}
              onClick={() => switchMode('login')}
              className={`py-2 rounded-lg transition ${
                mode === 'login'
                  ? 'bg-brand-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Sign In
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={mode === 'signup'}
              onClick={() => switchMode('signup')}
              className={`py-2 rounded-lg transition ${
                mode === 'signup'
                  ? 'bg-brand-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Create Account
            </button>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-4" noValidate>
          {!isConfigured && (
            <div
              role="alert"
              className="p-3 rounded-xl bg-amber-950/40 border border-amber-800/70 text-amber-200 text-xs flex items-start space-x-2"
            >
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-amber-400" />
              <span>
                Supabase frontend environment variables are not configured. Set{' '}
                <code className="font-mono">VITE_SUPABASE_URL</code> and{' '}
                <code className="font-mono">VITE_SUPABASE_ANON_KEY</code>.
              </span>
            </div>
          )}

          {confirmationMessage && (
            <div
              role="status"
              className="p-3 rounded-xl bg-emerald-950/40 border border-emerald-800/70 text-emerald-200 text-xs flex items-start space-x-2"
            >
              <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5 text-emerald-400" />
              <span>{confirmationMessage}</span>
            </div>
          )}

          {activeError && (
            <div
              role="alert"
              className="p-3 rounded-xl bg-rose-950/40 border border-rose-800/70 text-rose-200 text-xs flex items-start space-x-2"
            >
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-rose-400" />
              <span>{activeError}</span>
            </div>
          )}

          <div className="space-y-1.5">
            <label htmlFor="auth-email" className="block text-xs font-medium text-slate-300">
              Email Address
            </label>
            <div className="relative">
              <Mail className="w-4 h-4 text-slate-500 absolute left-3 top-2.5" />
              <input
                id="auth-email"
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                disabled={submitting}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-9 pr-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500"
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <label htmlFor="auth-password" className="block text-xs font-medium text-slate-300">
              Password
            </label>
            <div className="relative">
              <Lock className="w-4 h-4 text-slate-500 absolute left-3 top-2.5" />
              <input
                id="auth-password"
                type="password"
                autoComplete={mode === 'signup' ? 'new-password' : 'current-password'}
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder={mode === 'signup' ? 'At least 6 characters' : 'Enter your password'}
                disabled={submitting}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-9 pr-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500"
              />
            </div>
          </div>

          {mode === 'signup' && (
            <div className="space-y-1.5">
              <label htmlFor="auth-confirm-password" className="block text-xs font-medium text-slate-300">
                Confirm Password
              </label>
              <div className="relative">
                <Lock className="w-4 h-4 text-slate-500 absolute left-3 top-2.5" />
                <input
                  id="auth-confirm-password"
                  type="password"
                  autoComplete="new-password"
                  required
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="Re-enter your password"
                  disabled={submitting}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-9 pr-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500"
                />
              </div>
            </div>
          )}

          <button
            type="submit"
            disabled={submitting || !isConfigured}
            className="w-full py-2.5 px-4 rounded-xl bg-brand-600 hover:bg-brand-500 disabled:opacity-50 disabled:pointer-events-none text-white text-sm font-semibold transition flex items-center justify-center space-x-2 shadow-sm"
          >
            {submitting ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>{mode === 'signup' ? 'Creating account...' : 'Signing in...'}</span>
              </>
            ) : (
              <span>{mode === 'signup' ? 'Create Account' : 'Sign In'}</span>
            )}
          </button>
        </form>
      </div>
    </div>
  );
}
