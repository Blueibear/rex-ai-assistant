import React from 'react'
import { HashRouter, Routes, Route, NavLink } from 'react-router-dom'
import Dashboard from './pages/Dashboard.jsx'
import Chat from './pages/Chat.jsx'
import Voice from './pages/Voice.jsx'
import Settings from './pages/Settings.jsx'
import Logs from './pages/Logs.jsx'
import ShoppingList from './pages/ShoppingList.jsx'
import About from './pages/About.jsx'

const NAV_ITEMS = [
  { path: '/', label: 'Dashboard' },
  { path: '/chat', label: 'Chat' },
  { path: '/voice', label: 'Voice' },
  { path: '/settings', label: 'Settings' },
  { path: '/logs', label: 'Logs' },
  { path: '/shopping', label: 'Shopping List' },
  { path: '/about', label: 'About' },
]

const navStyle = {
  display: 'flex',
  gap: '1rem',
  padding: '0.75rem 1rem',
  background: '#1a1a2e',
  borderBottom: '1px solid #333',
}

const linkStyle = {
  color: '#ccc',
  textDecoration: 'none',
  padding: '0.25rem 0.5rem',
  borderRadius: '4px',
}

const activeLinkStyle = {
  ...linkStyle,
  color: '#fff',
  background: '#4a90e2',
}

const mainStyle = {
  padding: '1.5rem',
  fontFamily: 'system-ui, sans-serif',
  minHeight: '100vh',
  background: '#f5f5f5',
}

export default function App() {
  return (
    <HashRouter>
      <nav style={navStyle}>
        <span style={{ color: '#fff', fontWeight: 'bold', marginRight: '1rem' }}>Rex AI</span>
        {NAV_ITEMS.map(({ path, label }) => (
          <NavLink
            key={path}
            to={path}
            end={path === '/'}
            style={({ isActive }) => (isActive ? activeLinkStyle : linkStyle)}
          >
            {label}
          </NavLink>
        ))}
      </nav>
      <main style={mainStyle}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/chat" element={<Chat />} />
          <Route path="/voice" element={<Voice />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/logs" element={<Logs />} />
          <Route path="/shopping" element={<ShoppingList />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </main>
    </HashRouter>
  )
}
