import React, { useState } from 'react';
import { api } from '../lib/api.js';

export default function AuthForm({ mode }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState(null);
  const isLogin = mode === 'login';

  async function handleSubmit(e){
    e.preventDefault();
    setLoading(true); setMessage(null);
    try {
      const endpoint = isLogin ? '/api/auth/login' : '/api/auth/register';
      const { data } = await api.post(endpoint, { email, password });
      setMessage({ type: 'success', text: data.message || (isLogin ? 'Login success' : 'Registered') });
    } catch (err) {
      const msg = err.response?.data?.error || 'Request failed';
      setMessage({ type:'error', text: msg });
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} style={{display:'flex', flexDirection:'column', gap:'0.75rem'}}>
      <label style={labelStyle}>Email
        <input type="email" value={email} onChange={e=>setEmail(e.target.value)} required style={inputStyle} placeholder="you@example.com" />
      </label>
      <label style={labelStyle}>Password
        <input type="password" value={password} onChange={e=>setPassword(e.target.value)} required style={inputStyle} placeholder="••••••••" />
      </label>
      <button disabled={loading} style={{...buttonStyle, opacity:loading?0.7:1}}>{loading ? 'Please wait...' : (isLogin ? 'Login' : 'Register')}</button>
      {message && <div style={{
        padding:'0.5rem 0.75rem',
        borderRadius:6,
        background: message.type==='error' ? '#b91c1c' : '#065f46',
        fontSize:'0.8rem'
      }}>{message.text}</div>}
    </form>
  );
}

const labelStyle = {fontSize:'0.75rem', display:'flex', flexDirection:'column', gap:'0.35rem', fontWeight:600};
const inputStyle = {padding:'0.55rem 0.65rem', borderRadius:6, border:'1px solid #475569', background:'#0f172a', color:'#fff'};
const buttonStyle = {marginTop:'0.25rem', padding:'0.65rem 0.75rem', borderRadius:6, border:'none', cursor:'pointer', background:'#2563eb', color:'#fff', fontWeight:600};
