import React from 'react';
import './App.css';
// import logo from './logo.svg';
import SymptomForm from './components/SymptomForm';
import Chatbot from './components/ChatBot';

function Navbar() {
  return (
    <nav className="navbar dark">
      <div className="navbar-logo">
        <img src={process.env.PUBLIC_URL + '/smartdoc.png'} alt="Smart Doctor Logo" style={{ width: 40, height: 40, borderRadius: 8 }} />
        <span className="navbar-title">Smart Doctor</span>
      </div>
    </nav>
  );
}

function Sidebar() {
  return (
    <aside className="sidebar dark">
      <Chatbot darkMode={true} />
    </aside>
  );
}

function MainContent() {
  return (
    <section className="main-content dark">
      <SymptomForm />
    </section>
  );
}

function App() {
  return (
    <div className="App dark">
      <Navbar />
      <div className="container">
        <Sidebar />
        <MainContent />
      </div>
    </div>
  );
}

export default App;
