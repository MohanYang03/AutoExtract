import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, Navigate, useLocation } from 'react-router-dom';
import BatchExtractor from './components/BatchExtractor';
import 'bootstrap/dist/css/bootstrap.min.css';

const HomePage = () => (
  <div className="home-container">
    <div className="card border-0 shadow-sm mb-5">
      <div className="card-body text-center p-5">
        <h1 className="hero-title mb-4">
          <span className="gradient-text">AutoExtract</span>
        </h1>
        <h2 className="hero-subtitle mb-4">AI-Powered Data Extraction</h2>
        <p className="hero-description mb-4">
          Transform charts, graphs, and visual data into structured, analyzable information
          using cutting-edge AI detection and interactive tools.
        </p>
        <div className="hero-actions">
          <Link to="/bar_analyzer" className="btn btn-primary me-3">
            <i className="bi bi-graph-up me-2"></i>Bar Analyzer
          </Link>
          <Link to="/box_analyzer" className="btn btn-outline-primary me-3">
            <i className="bi bi-box me-2"></i>Box Analyzer
          </Link>
          <Link to="/violin_analyzer" className="btn btn-outline-primary me-3">
            <i className="bi bi-vinyl me-2"></i>Violin Analyzer
          </Link>
          <Link to="/line_analyzer" className="btn btn-outline-primary">
            <i className="bi bi-graph-up me-2"></i>Line Analyzer
          </Link>
        </div>
      </div>
    </div>

    <div className="features-section">
      <div className="row g-4">
        {[
          { icon: 'bi-graph-up-arrow', title: 'Bar Graph Analysis', desc: 'Drop in whole papers: every bar chart and dot plot is measured (means and error bars) and exported in long format.', link: '/bar_analyzer' },
          { icon: 'bi-box', title: 'Box Plot Analysis', desc: 'Read min, quartiles, median and max from box plots and estimate mean and SD for meta-analysis (Luo, Shi, Wan, Hozo).', link: '/box_analyzer' },
          { icon: 'bi-vinyl', title: 'Violin Plot Analysis', desc: 'Read the ends, quartile and median lines (or an inner box plot) of violin plots and estimate mean and SD.', link: '/violin_analyzer' },
          { icon: 'bi-graph-up', title: 'Line Graph Analysis', desc: 'Trace each series of a line plot: every point with its error bar, and the x value from the axis.', link: '/line_analyzer' }
        ].map((feature, i) => (
          <div key={i} className="col-lg-3 col-md-6">
            <div className="card border-0 shadow-sm h-100">
              <div className="card-body text-center">
                <div className="feature-icon-compact mb-3">
                  <i className={`bi ${feature.icon}`}></i>
                </div>
                <h5 className="mb-3">{feature.title}</h5>
                <p className="mb-4 text-muted">{feature.desc}</p>
                <Link to={feature.link} className="btn btn-outline-primary btn-sm">
                  Learn More <i className="bi bi-arrow-right"></i>
                </Link>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  </div>
);

// One page per chart type. Both stay mounted once visited, so dropped papers
// and edits are kept while you switch pages; only the current one is shown.
const ANALYZER_PATHS = {
  '/bar_analyzer': 'bar', '/box_analyzer': 'box', '/violin_analyzer': 'violin', '/line_analyzer': 'line',
};

const AnalyzerPages = () => {
  const { pathname } = useLocation();
  const current = ANALYZER_PATHS[pathname.replace(/\/+$/, '')];
  const [visited, setVisited] = useState({});
  if (current && !visited[current]) setVisited((v) => ({ ...v, [current]: true }));
  return ['bar', 'box', 'violin', 'line'].map((mode) => (visited[mode] || current === mode) && (
    <div key={mode} style={{ display: current === mode ? 'block' : 'none' }}>
      <BatchExtractor mode={mode} />
    </div>
  ));
};

function App() {
  return (
    <Router>
      <nav className="navbar navbar-expand-lg navbar-light bg-light">
        <div className="container-fluid content mt-4">
          <Link className="navbar-brand" to="/">AutoExtract</Link>
          <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav"
            aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
            <span className="navbar-toggler-icon"></span>
          </button>
          <div className="collapse navbar-collapse" id="navbarNav">
            <ul className="navbar-nav">
              <li className="nav-item"><Link className="nav-link" to="/">Home</Link></li>
              <li className="nav-item"><Link className="nav-link" to="/bar_analyzer">Bar Analyzer</Link></li>
              <li className="nav-item"><Link className="nav-link" to="/box_analyzer">Box Analyzer</Link></li>
              <li className="nav-item"><Link className="nav-link" to="/violin_analyzer">Violin Analyzer</Link></li>
              <li className="nav-item"><Link className="nav-link" to="/line_analyzer">Line Analyzer</Link></li>
            </ul>
          </div>
        </div>
      </nav>

      <div className="container-fluid content mt-4">
        <Routes>
          <Route path="/bar_analyzer" element={null} />
          <Route path="/box_analyzer" element={null} />
          <Route path="/violin_analyzer" element={null} />
          <Route path="/line_analyzer" element={null} />
          <Route path="/extract" element={<Navigate to="/bar_analyzer" replace />} />
          <Route path="/tutorial" element={<Navigate to="/" replace />} />
          <Route path="/" element={<HomePage />} />
        </Routes>
        <AnalyzerPages />
      </div>

      <footer className="bg-light text-center py-2 mt-4" style={{
        position: "fixed",
        bottom: 0,
        left: 0,
        backgroundColor: "white",
        padding: "8px",
        borderTopRightRadius: "10px",
        boxShadow: "2px -2px 5px rgba(0, 0, 0, 0.1)"
      }}>
        <p className="mb-0 small">
          <a href="https://xulab.anat.uci.edu/" target="_blank" rel="noopener noreferrer" className="me-3">Xulab</a>
          <a href="https://ics.uci.edu/~zhaoxia/" target="_blank" rel="noopener noreferrer" className="me-3">Yulab</a>
          <a href="https://github.com/MohanYang03/AutoExtract" target="_blank" rel="noopener noreferrer">GitHub</a>
        </p>
      </footer>
    </Router>
  );
}

export default App;
