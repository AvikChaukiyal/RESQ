import React from 'react';
import { Routes, Route } from 'react-router-dom';
import './App.css';
import UserSignUp from './components/Auth/UserSignUp';
import AuthoritySignUp from './components/Auth/AuthoritySignUp';
import UserLogin from './components/Auth/UserLogin';
import AuthorityLogin from './components/Auth/AuthorityLogin';
import UserDashboard from './components/User/UserDashboard';
import AuthorityDashboard from './components/Authority/AuthorityDashboard';
import EmergencyWarning from './components/User/EmergencyWarning';
import RescueCampMap from './components/User/RescueCampMap';
import UserProfile from './components/User/UserProfile';
import ReportIncident from './components/User/ReportIncident';
import Terms from './components/Shared/Terms';
import Camps from './components/Authority/Camps';
import Incidents from './components/Authority/Incidents';
import UsersOnMap from './components/Authority/UsersOnMap';
import Alerts from './components/Authority/Alerts';
import Rations from './components/Authority/Rations';
import Users from './components/Authority/Users';
import Reports from './components/Authority/Reports';
import Settings from './components/Authority/Settings';
import AdminLayout from './components/Shared/AdminLayout';
import Home from './components/Authority/Home';
import ProtectedRoute from './components/Shared/ProtectedRoute';
import ChatbotPage from './components/Shared/ChatbotPage';
import DisasterHistory from './components/Authority/DisasterHistory';

function App() {
  return (
    <div className="App">
      <Routes>
        <Route path="/" element={<UserLogin />} />
        <Route path="/login/user" element={<UserLogin />} />
        <Route path="/signup/user" element={<UserSignUp />} />
        <Route path="/signup/authority" element={<AuthoritySignUp />} />
        <Route path="/login/authority" element={<AuthorityLogin />} />
        <Route path="/user/dashboard" element={<ProtectedRoute><UserDashboard /></ProtectedRoute>} />
        <Route path="/authority/dashboard" element={<ProtectedRoute><AdminLayout><Home /></AdminLayout></ProtectedRoute>} />
        <Route path="/user/warnings" element={<ProtectedRoute><EmergencyWarning /></ProtectedRoute>} />
        <Route path="/user/map" element={<ProtectedRoute><RescueCampMap /></ProtectedRoute>} />
        <Route path="/user/profile" element={<ProtectedRoute><UserProfile /></ProtectedRoute>} />
        <Route path="/user/incidents" element={<ProtectedRoute><ReportIncident /></ProtectedRoute>} />
        <Route path="/terms" element={<Terms />} />
        <Route path="/authority/camps" element={<ProtectedRoute><AdminLayout><Camps /></AdminLayout></ProtectedRoute>} />
        <Route path="/authority/incidents" element={<ProtectedRoute><AdminLayout><Incidents /></AdminLayout></ProtectedRoute>} />
        <Route path="/authority/users-on-map" element={<ProtectedRoute><UsersOnMap /></ProtectedRoute>} />
        <Route path="/authority/alerts" element={<ProtectedRoute><AdminLayout><Alerts /></AdminLayout></ProtectedRoute>} />
        <Route path="/authority/rations" element={<ProtectedRoute><Rations /></ProtectedRoute>} />
        <Route path="/authority/users" element={<ProtectedRoute><Users /></ProtectedRoute>} />
        <Route path="/authority/reports" element={<ProtectedRoute><Reports /></ProtectedRoute>} />
        <Route path="/authority/settings" element={<ProtectedRoute><Settings /></ProtectedRoute>} />
        <Route path="/user/chatbot" element={<ProtectedRoute><ChatbotPage /></ProtectedRoute>} />
        <Route path="/authority/chatbot" element={<ProtectedRoute><ChatbotPage /></ProtectedRoute>} />
        <Route path="/authority/disaster-history" element={<ProtectedRoute><DisasterHistory /></ProtectedRoute>} />
      </Routes>
    </div>
  );
}

export default App;
