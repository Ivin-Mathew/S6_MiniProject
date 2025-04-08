import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import './index.css'
import App from './App.jsx'
import Preview from './Preview.jsx';
import PreviewV2 from './components/PreviewV2.jsx';

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <Router>
    <Routes>
      <Route path="/" element={<App />} />
      <Route path="/preview" element={<PreviewV2 />} />
    </Routes>
  </Router>
  </StrictMode>,
)
