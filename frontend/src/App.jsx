import React, { useState } from 'react';
import AuthForm from './components/AuthForm.jsx';

export default function App() {
  const [mode, setMode] = useState('login');
  return (
    <div style={{
      fontFamily: 'system-ui, sans-serif',
      display: 'flex',
      minHeight: '100vh',
      alignItems: 'center',
      justifyContent: 'center',
      background: '#0f172a',
      color: '#fff'
    }}>
      <div style={{background:'#1e293b', padding:'2rem', borderRadius: '12px', width:'340px', boxShadow:'0 8px 24px -8px rgba(0,0,0,0.5)'}}>
        <h1 style={{marginTop:0, fontSize:'1.4rem', textAlign:'center'}}>Auth Portal</h1>
        <div style={{display:'flex', gap:'0.5rem', marginBottom:'1rem'}}>
          <button onClick={()=>setMode('login')} style={btnStyle(mode==='login')}>Login</button>
          <button onClick={()=>setMode('register')} style={btnStyle(mode==='register')}>Register</button>
        </div>
        <AuthForm mode={mode} />
        <p style={{fontSize:'0.7rem', opacity:0.6, marginTop:'1.5rem', textAlign:'center'}}>Flask + React + Postgres Demo</p>
      </div>
    </div>
  );
}

function btnStyle(active){
  return {
    flex:1,
    padding:'0.55rem 0.75rem',
    background: active ? '#3b82f6' : '#334155',
    color:'#fff',
    border:'none',
    borderRadius:6,
    cursor:'pointer',
    fontWeight:600
  }
}
