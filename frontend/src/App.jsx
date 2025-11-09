import { useState } from 'react'
import { Navigate, Route, Routes, useNavigate } from 'react-router-dom'
import HomePage from './components/HomePage'
import LiveWebcamShell from './pages/LiveWebcamShell'

function HomeRoute() {
  const navigate = useNavigate()
  return <HomePage onBegin={() => navigate('/live')} hasSummary={false} />
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<HomeRoute />} />
      <Route path="/live" element={<LiveWebcamShell />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

