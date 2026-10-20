import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import BarAnalyzer from './components/BarAnalyzer';
import BatchExtractor from './components/BatchExtractor';
import 'bootstrap/dist/css/bootstrap.min.css';



// Video Component
const TutorialVideo = ({ videoUrl, title }) => {
  const [videoError, setVideoError] = useState(false);

  if (!videoUrl) {
    return (
      <div className="card border-0 shadow-sm mb-4">
        <div className="card-body p-5">
          <div className="text-center">
            <i className="bi bi-camera-video display-1 text-muted"></i>
            <h6 className="text-muted mt-3">Video Coming Soon</h6>
            <p className="text-muted">Demonstration video for: <strong>{title}</strong></p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="card border-0 shadow-sm mb-4">
      <div className="card-body p-0">
        {videoError ? (
          <div className="p-5 text-center">
            <i className="bi bi-exclamation-triangle display-1 text-warning"></i>
            <h6 className="text-warning mt-3">Video Unavailable</h6>
            <p className="text-muted">The video for this step is currently unavailable.</p>
          </div>
        ) : (
          <video
            autoPlay
            muted
            loop
            playsInline
            className="w-100 rounded"
            onError={() => setVideoError(true)}
          >
            <source src={videoUrl} type="video/mp4" />
          </video>
        )}
      </div>
    </div>
  );
};

// Home Page Component
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
          <Link to="/tutorial" className="btn btn-primary me-3">
            <i className="bi bi-play-circle me-2"></i>Start Learning
          </Link>
          <Link to="/extract" className="btn btn-primary me-3">
            <i className="bi bi-files me-2"></i>Batch Extract Papers
          </Link>
          <Link to="/bar_analyzer" className="btn btn-outline-primary">
            <i className="bi bi-graph-up me-2"></i>Try Bar Analyzer
          </Link>
        </div>
      </div>
    </div>

    <div className="features-section">
      <div className="row g-4">
        {[
          { icon: 'bi-graph-up-arrow', title: 'Bar Graph Analysis', desc: 'Extract precise measurements from bar charts with AI-powered detection and interactive editing tools.', link: '/bar_analyzer' },
          { icon: 'bi-box', title: 'Box Detection', desc: 'Identify and analyze rectangular elements in images with advanced computer vision algorithms.', link: '/box_analyzer' },
          { icon: 'bi-graph-up', title: 'Line Graph Analysis', desc: 'Extract data points and trends from line graphs with precision and accuracy.', link: '/line_analyzer' }
        ].map((feature, i) => (
          <div key={i} className="col-lg-4 col-md-6">
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

// Tutorial Page Component
const TutorialPage = () => {
  const [activeStep, setActiveStep] = useState(0);

  const tutorialSteps = [
    {
      id: 0,
      title: "Welcome to Bar Graph Analyzer",
      description: "Learn how to extract data from bar charts using AI-powered detection and interactive tools.",
      videoUrl: "/videos/step1.mov",
      content: (
        <div className="card border-0 shadow-sm mb-4">
          <div className="card-header bg-primary text-white">
            <h6 className="mb-0">🎯 What You'll Learn</h6>
          </div>
          <div className="card-body">
            <div className="row g-3">
              {[
                { icon: '📤', title: 'Upload & Analyze', desc: 'Upload bar chart images and let AI detect chart elements automatically' },
                { icon: '✏️', title: 'Interactive Editing', desc: 'Adjust detection boxes, change categories, and fine-tune positions' },
                { icon: '📊', title: 'Data Extraction', desc: 'Calculate bar heights, export to CSV/Excel, and get structured data' }
              ].map((feature, i) => (
                <div key={i} className="col-md-4">
                  <div className="card border-0 shadow-sm h-100">
                    <div className="card-body text-center">
                      <h6 className="mb-2">{feature.icon} {feature.title}</h6>
                      <p className="small text-muted">{feature.desc}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )
    },
    {
      id: 1,
      title: "Step 1: Upload Your Bar Chart",
      description: "Start by uploading a clear image of your bar chart.",
      videoUrl: "",
      content: (
        <div className="card border-0 shadow-sm mb-4">
          <div className="card-header bg-info text-white">
            <h6 className="mb-0">📤 Upload Your Image</h6>
          </div>
          <div className="card-body">
            <div className="alert alert-info mb-4">
              <strong>Requirements:</strong>
              <ul className="mb-0 mt-2">
                <li>Clear, high-resolution image of a bar chart</li>
                <li>Supported formats: PNG, JPG, JPEG</li>
                <li>Chart should have visible bars, axes, and labels</li>
              </ul>
            </div>

            <div className="row g-3">
              <div className="col-md-6">
                <div className="card border-0 shadow-sm">
                  <div className="card-header bg-light">
                    <h6 className="mb-0">✅ What Works Best</h6>
                  </div>
                  <div className="card-body">
                    <ul className="mb-0">
                      <li>Charts with clear contrast between bars and background</li>
                      <li>Well-defined axes and tick marks</li>
                      <li>Readable text labels</li>
                      <li>Bars that are clearly separated</li>
                    </ul>
                  </div>
                </div>
              </div>
              <div className="col-md-6">
                <div className="card border-0 shadow-sm">
                  <div className="card-header bg-light">
                    <h6 className="mb-0">❌ What to Avoid</h6>
                  </div>
                  <div className="card-body">
                    <ul className="mb-0">
                      <li>Blurry or low-resolution images</li>
                      <li>Charts with overlapping elements</li>
                      <li>Very small text that's hard to read</li>
                      <li>Complex 3D or stacked bar charts</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            <div className="alert alert-warning mt-4">
              <strong>💡 Pro Tip:</strong> Take a screenshot or use a scanner for best results. The clearer the image, the better the AI detection will be!
            </div>
          </div>
        </div>
      )
    },
    {
      id: 2,
      title: "Step 2: AI Detection & Analysis",
      description: "Watch as AI automatically detects chart elements and analyzes your data.",
      videoUrl: "/videos/ai-detection.mp4",
      content: (
        <div className="card border-0 shadow-sm mb-4">
          <div className="card-header bg-success text-white">
            <h6 className="mb-0">🤖 AI-Powered Detection</h6>
          </div>
          <div className="card-body">
            <p className="mb-4">Once you upload your image, our AI will automatically:</p>

            <div className="row g-3">
              <div className="col-md-6">
                <div className="card border-0 shadow-sm">
                  <div className="card-header bg-light">
                    <h6 className="mb-0">🔍 What Gets Detected</h6>
                  </div>
                  <div className="card-body">
                    <div className="list-group list-group-flush">
                      {[
                        { icon: '📊', title: 'Bar Elements', desc: 'Individual bars in your chart' },
                        { icon: '📏', title: 'Axes', desc: 'X and Y axis lines and labels' },
                        { icon: '🏷️', title: 'Chart Labels', desc: 'Title, axis labels, and tick marks' },
                        { icon: '📈', title: 'Data Points', desc: 'Error bars, data markers, etc.' }
                      ].map((item, i) => (
                        <div key={i} className="list-group-item border-0 px-0">
                          <h6 className="mb-1">{item.icon} {item.title}</h6>
                          <small className="text-muted">{item.desc}</small>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              <div className="col-md-6">
                <div className="card border-0 shadow-sm">
                  <div className="card-header bg-light">
                    <h6 className="mb-0">⚙️ Detection Process</h6>
                  </div>
                  <div className="card-body">
                    <ol className="mb-0">
                      <li><strong>Image Analysis:</strong> AI scans your chart image</li>
                      <li><strong>Element Recognition:</strong> Identifies bars, axes, and labels</li>
                      <li><strong>OCR Processing:</strong> Reads text and numerical values</li>
                      <li><strong>Coordinate Mapping:</strong> Maps positions of all elements</li>
                      <li><strong>Data Extraction:</strong> Calculates bar heights and positions</li>
                    </ol>
                  </div>
                </div>
              </div>
            </div>

            <div className="alert alert-success mt-4">
              <strong>🎯 Expected Outcome:</strong> You'll see green bounding boxes around detected elements, with the AI attempting to read axis values and chart labels.
            </div>
          </div>
        </div>
      )
    },
    {
      id: 3,
      title: "Step 3: Interactive Detection Viewer",
      description: "Use the interactive tools to fine-tune AI detections and add missing elements.",
      videoUrl: "/videos/interactive-editing.mp4",
      content: (
        <div className="card border-0 shadow-sm mb-4">
          <div className="card-header bg-primary text-white">
            <h6 className="mb-0">✏️ Interactive Detection Viewer</h6>
          </div>
          <div className="card-body">
            <p className="mb-4">This is where you can perfect the AI's work and ensure accurate data extraction.</p>

            <div className="row g-3">
              <div className="col-md-6">
                <div className="card border-0 shadow-sm">
                  <div className="card-header bg-light">
                    <h6 className="mb-0">🖱️ Selection & Editing</h6>
                  </div>
                  <div className="card-body">
                    <ul className="mb-0">
                      <li><strong>Click on any box</strong> to select and see its details</li>
                      <li><strong>Drag boxes</strong> to adjust their positions</li>
                      <li><strong>Resize boxes</strong> using the corner handles</li>
                      <li><strong>Change categories</strong> using the dropdown menu</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="col-md-6">
                <div className="card border-0 shadow-sm">
                  <div className="card-header bg-light">
                    <h6 className="mb-0">➕ Adding New Elements</h6>
                  </div>
                  <div className="card-body">
                    <ul className="mb-0">
                      <li><strong>Add Box button</strong> creates new detection boxes</li>
                      <li><strong>Customize position</strong> and size as needed</li>
                      <li><strong>Set category</strong> (bar, axis, label, etc.)</li>
                      <li><strong>Sync changes</strong> to save modifications</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            <div className="alert alert-info mt-4">
              <h6 className="mb-3">🔧 Key Tools</h6>
              <div className="row g-3">
                {[
                  { title: 'Selection Info Panel', desc: 'View and edit selected box properties' },
                  { title: 'Value Editor', desc: 'Set Y-axis origin and maximum values' },
                  { title: 'Sync Button', desc: 'Save all changes to the backend' },
                  { title: 'Debug Button', desc: 'Check system state and troubleshoot' }
                ].map((tool, i) => (
                  <div key={i} className="col-md-3">
                    <div className="card border-0 shadow-sm">
                      <div className="card-body text-center">
                        <strong className="d-block mb-1">{tool.title}</strong>
                        <small className="text-muted">{tool.desc}</small>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 4,
      title: "Step 4: Fine-tune Detection Boxes",
      description: "Learn how to adjust detection boxes for perfect accuracy.",
      videoUrl: "/videos/fine-tuning.mp4",
      content: (
        <div className="card border-0 shadow-sm mb-4">
          <div className="card-header bg-primary text-white">
            <h6 className="mb-0">🎯 Fine-tuning Detection Boxes</h6>
          </div>
          <div className="card-body">
            <p className="mb-4">Perfect the AI's detection by adjusting boxes to precisely match your chart elements.</p>

            <div className="row g-3">
              <div className="col-md-6">
                <div className="card border-0 shadow-sm">
                  <div className="card-header bg-light">
                    <h6 className="mb-0">📏 Box Manipulation</h6>
                  </div>
                  <div className="card-body">
                    <h6>Resize Handles:</h6>
                    <ul className="mb-3">
                      <li><strong>Corner handles:</strong> Resize proportionally</li>
                      <li><strong>Edge handles:</strong> Resize horizontally/vertically</li>
                      <li><strong>Rotation handle:</strong> Rotate the box (top center)</li>
                    </ul>
                    <small className="text-muted">💡 Handles appear when you select a box</small>
                  </div>
                </div>
              </div>

              <div className="col-md-6">
                <div className="card border-0 shadow-sm">
                  <div className="card-header bg-light">
                    <h6 className="mb-0">🏷️ Category Management</h6>
                  </div>
                  <div className="card-body">
                    <h6>Available Categories:</h6>
                    <ul className="mb-3">
                      <li><strong>bar:</strong> Individual bar elements</li>
                      <li><strong>yaxis:</strong> Y-axis line and labels</li>
                      <li><strong>xaxis:</strong> X-axis line and labels</li>
                      <li><strong>label:</strong> Chart title and text</li>
                    </ul>
                    <small className="text-muted">💡 Change categories using the dropdown menu</small>
                  </div>
                </div>
              </div>
            </div>

            <div className="alert alert-warning mt-4">
              <h6 className="mb-3">⚠️ Important Notes</h6>
              <ul className="mb-0">
                <li><strong>Always click "Sync"</strong> after making changes to save them</li>
                <li><strong>Boxes must not overlap</strong> for accurate detection</li>
                <li><strong>Ensure proper categorization</strong> for correct height calculations</li>
                <li><strong>Use the Debug button</strong> if you encounter issues</li>
              </ul>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 5,
      title: "Step 5: Set Chart Reference Values",
      description: "Configure Y-axis origin and maximum values for accurate height calculations.",
      videoUrl: "/videos/reference-values.mp4",
      content: (
        <div className="card border-0 shadow-sm mb-4">
          <div className="card-header bg-primary text-white">
            <h6 className="mb-0">📊 Chart Reference Values</h6>
          </div>
          <div className="card-body">
            <p className="mb-4">Set the Y-axis scale to convert pixel measurements to actual data values.</p>

            <div className="row g-3">
              <div className="col-md-6">
                <div className="card border-0 shadow-sm">
                  <div className="card-header bg-light">
                    <h6 className="mb-0">🔢 Required Values</h6>
                  </div>
                  <div className="card-body">
                    <h6>Y-axis Origin (Y=0):</h6>
                    <p className="mb-3">The value at the bottom of your Y-axis (usually 0)</p>
                    <h6>Y-axis Maximum:</h6>
                    <p className="mb-3">The highest value on your Y-axis</p>
                    <small className="text-muted">💡 These values are used to convert pixel heights to data values</small>
                  </div>
                </div>
              </div>

              <div className="col-md-6">
                <div className="card border-0 shadow-sm">
                  <div className="card-header bg-light">
                    <h6 className="mb-0">📐 How to Set Values</h6>
                  </div>
                  <div className="card-body">
                    <ol className="mb-0">
                      <li><strong>Check the Value Editor panel</strong> above the canvas</li>
                      <li><strong>Enter the origin value</strong> (e.g., 0)</li>
                      <li><strong>Enter the maximum value</strong> (e.g., 100)</li>
                      <li><strong>Click "Update Values"</strong> to save</li>
                      <li><strong>Verify the values</strong> are correctly displayed</li>
                    </ol>
                  </div>
                </div>
              </div>
            </div>

            <div className="alert alert-info mt-4">
              <h6 className="mb-3">💡 Pro Tips</h6>
              <ul className="mb-0">
                <li><strong>Use exact values</strong> from your chart's Y-axis</li>
                <li><strong>Check for units</strong> (%, kg, etc.) and include them</li>
                <li><strong>Verify with chart labels</strong> to ensure accuracy</li>
                <li><strong>Test with known values</strong> if possible</li>
              </ul>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 6,
      title: "Step 6: Calculate Bar Heights",
      description: "Generate accurate measurements and extract your data.",
      videoUrl: "/videos/calculate-heights.mp4",
      content: (
        <div className="card border-0 shadow-sm mb-4">
          <div className="card-header bg-primary text-white">
            <h6 className="mb-0">📏 Calculate Bar Heights</h6>
          </div>
          <div className="card-body">
            <p className="mb-4">Once all elements are properly detected and values are set, calculate the actual bar heights.</p>

            <div className="row g-3">
              <div className="col-md-6">
                <div className="card border-0 shadow-sm">
                  <div className="card-header bg-light">
                    <h6 className="mb-0">✅ Prerequisites Check</h6>
                  </div>
                  <div className="card-body">
                    <h6>Required Elements:</h6>
                    <ul className="mb-3">
                      <li>✅ At least one <strong>bar</strong> box</li>
                      <li>✅ <strong>yaxis</strong> box detected</li>
                      <li>✅ <strong>xaxis</strong> box detected</li>
                      <li>✅ <strong>Origin value</strong> set</li>
                      <li>✅ <strong>Maximum value</strong> set</li>
                    </ul>
                    <small className="text-muted">💡 The "Calculate Heights" button will be enabled when all requirements are met</small>
                  </div>
                </div>
              </div>

              <div className="col-md-6">
                <div className="card border-0 shadow-sm">
                  <div className="card-header bg-light">
                    <h6 className="mb-0">🚀 Calculation Process</h6>
                  </div>
                  <div className="card-body">
                    <ol className="mb-0">
                      <li><strong>Click "Calculate Heights"</strong> button</li>
                      <li><strong>AI processes</strong> all detection boxes</li>
                      <li><strong>Converts pixel measurements</strong> to data values</li>
                      <li><strong>Calculates bar heights</strong> and error bars</li>
                      <li><strong>Displays results</strong> in organized format</li>
                    </ol>
                  </div>
                </div>
              </div>
            </div>

            <div className="alert alert-success mt-4">
              <h6 className="mb-3">🎯 Expected Results</h6>
              <ul className="mb-0">
                <li><strong>Bar heights</strong> in your specified units</li>
                <li><strong>Error bar values</strong> if detected</li>
                <li><strong>Bar labels</strong> (customizable)</li>
                <li><strong>Chart title</strong> extracted from image</li>
              </ul>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 7,
      title: "Step 7: Export Your Data",
      description: "Download your extracted data in CSV or Excel format.",
      videoUrl: "/videos/export-data.mp4",
      content: (
        <div className="card border-0 shadow-sm mb-4">
          <div className="card-header bg-primary text-white">
            <h6 className="mb-0">📤 Export Your Data</h6>
          </div>
          <div className="card-body">
            <p className="mb-4">Download your extracted data in multiple formats for further analysis.</p>

            <div className="row g-3">
              <div className="col-md-6">
                <div className="card border-0 shadow-sm">
                  <div className="card-header bg-light">
                    <h6 className="mb-0">📊 Export Formats</h6>
                  </div>
                  <div className="card-body">
                    <h6>CSV Export:</h6>
                    <p className="mb-3">Comma-separated values, perfect for spreadsheet applications</p>
                    <h6>Excel Export:</h6>
                    <p className="mb-3">Native Excel format (.xlsx) with proper formatting</p>
                    <small className="text-muted">💡 Both formats contain the same data in long format</small>
                  </div>
                </div>
              </div>

              <div className="col-md-6">
                <div className="card border-0 shadow-sm">
                  <div className="card-header bg-light">
                    <h6 className="mb-0">📋 Data Structure</h6>
                  </div>
                  <div className="card-body">
                    <h6>Columns:</h6>
                    <ul className="mb-3">
                      <li><strong>mean:</strong> Bar height values</li>
                      <li><strong>error:</strong> Error bar values (if any)</li>
                      <li><strong>group:</strong> Bar labels/names</li>
                    </ul>
                    <small className="text-muted">💡 Each row represents one bar from your chart</small>
                  </div>
                </div>
              </div>
            </div>

            <div className="alert alert-info mt-4">
              <h6 className="mb-3">🔧 Export Options</h6>
              <ul className="mb-0">
                <li><strong>Filename:</strong> Automatically uses chart title or "bar_analysis_results"</li>
                <li><strong>Data Preview:</strong> Shows first 2 rows before export</li>
                <li><strong>Format Selection:</strong> Choose between CSV and Excel</li>
                <li><strong>Download Location:</strong> Saves to your default downloads folder</li>
              </ul>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 8,
      title: "Troubleshooting & Tips",
      description: "Common issues and solutions to help you succeed.",
      videoUrl: "/videos/troubleshooting.mp4",
      content: (
        <div className="card border-0 shadow-sm mb-4">
          <div className="card-header bg-primary text-white">
            <h6 className="mb-0">🔧 Troubleshooting & Tips</h6>
          </div>
          <div className="card-body">
            <p className="mb-4">Solutions to common issues and best practices for optimal results.</p>

            <div className="row g-3">
              <div className="col-md-6">
                <div className="card border-0 shadow-sm">
                  <div className="card-header bg-light">
                    <h6 className="mb-0">🚨 Common Issues</h6>
                  </div>
                  <div className="card-body">
                    <h6>Detection Problems:</h6>
                    <ul className="mb-3">
                      <li><strong>Missing elements:</strong> Add boxes manually using "Add Box"</li>
                      <li><strong>Wrong categories:</strong> Use dropdown to change box types</li>
                      <li><strong>Poor accuracy:</strong> Ensure high-quality, clear images</li>
                      <li><strong>Overlapping boxes:</strong> Adjust positions to prevent overlap</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="col-md-6">
                <div className="card border-0 shadow-sm">
                  <div className="card-header bg-light">
                    <h6 className="mb-0">💡 Best Practices</h6>
                  </div>
                  <div className="card-body">
                    <h6>For Best Results:</h6>
                    <ul className="mb-3">
                      <li><strong>Use clear images</strong> with good contrast</li>
                      <li><strong>Verify all required elements</strong> are detected</li>
                      <li><strong>Double-check values</strong> before calculating</li>
                      <li><strong>Sync changes frequently</strong> to avoid losing work</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            <div className="alert alert-warning mt-4">
              <h6 className="mb-3">⚠️ Important Reminders</h6>
              <ul className="mb-0">
                <li><strong>Always click "Sync"</strong> after making changes</li>
                <li><strong>Use the Debug button</strong> if you encounter issues</li>
                <li><strong>Check the console</strong> for detailed error messages</li>
                <li><strong>Verify box categories</strong> are correct before calculating</li>
              </ul>
            </div>

            <div className="text-center mt-4">
              <Link to="/bar_analyzer" className="btn btn-primary">
                🚀 Start Using Bar Analyzer
              </Link>
            </div>
          </div>
        </div>
      )
    }
  ];

  return (
    <div className="tutorial-container">
      <div className="row">
        {/* Left Sidebar: Step Navigation */}
        <div className="col-md-3">
          <div className="card border-0 shadow-sm mb-4">
            <div className="card-header bg-primary text-white">
              <h6 className="mb-0">
                <i className="bi bi-list-ol me-2"></i>Tutorial Steps
              </h6>
            </div>
            <div className="card-body">
              <div className="d-flex flex-column gap-2">
                {tutorialSteps.map((step, index) => (
                  <button
                    key={step.id}
                    className={`btn ${index === activeStep ? 'btn-primary' : 'btn-outline-secondary'} btn-sm text-start`}
                    onClick={() => setActiveStep(index)}
                  >
                    <span className="step-number me-2">{index + 1}.</span>
                    <span className="step-title">{step.title.split(':')[1] || step.title}</span>
                  </button>
                ))}
              </div>

              <div className="mt-4">
                <div className="progress mb-2">
                  <div
                    className="progress-bar"
                    role="progressbar"
                    style={{ width: `${((activeStep + 1) / tutorialSteps.length) * 100}%` }}
                    aria-valuenow={activeStep + 1}
                    aria-valuemin="0"
                    aria-valuemax={tutorialSteps.length}
                  ></div>
                </div>
                <div className="text-center small text-muted">
                  Step {activeStep + 1} of {tutorialSteps.length}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Content Area */}
        <div className="col-md-9">
          <div className="video-section mb-3">
            <TutorialVideo
              videoUrl={tutorialSteps[activeStep].videoUrl}
              title={tutorialSteps[activeStep].title}
            />
          </div>

          <div className="step-content mb-4">
            <div className="card border-0 shadow-sm">
              <div className="card-header bg-primary text-white">
                <h5 className="mb-0">
                  <i className="bi bi-lightbulb me-2"></i>
                  {tutorialSteps[activeStep].title}
                </h5>
              </div>
              <div className="card-body">
                <div className="step-details">
                  {tutorialSteps[activeStep].content}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>


    </div>
  );
};

function App() {
  return (
    <Router>
      {/* Navigation Bar */}
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
              <li className="nav-item"><Link className="nav-link" to="/tutorial">Tutorial</Link></li>
              <li className="nav-item"><Link className="nav-link" to="/extract">Batch Extract</Link></li>
              <li className="nav-item"><Link className="nav-link" to="/bar_analyzer">Bar Analyzer</Link></li>
              <li className="nav-item"><Link className="nav-link" to="/box_analyzer">Box Analyzer</Link></li>
              <li className="nav-item"><Link className="nav-link" to="/line_analyzer">Line Analyzer</Link></li>
            </ul>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="container-fluid content mt-4">
        <Routes>
          <Route path="/bar_analyzer" element={<BarAnalyzer />} />
          <Route path="/extract" element={<BatchExtractor />} />
          <Route path="/tutorial" element={<TutorialPage />} />
          <Route path="/" element={<HomePage />} />
        </Routes>
      </div>

      {/* Footer */}
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
          <a href="https://github.com/HenryYang03/AutoExtract" target="_blank" rel="noopener noreferrer">GitHub</a>
        </p>
      </footer>
    </Router>
  );
}

export default App;