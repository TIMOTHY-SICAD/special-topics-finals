import React, { useRef, useState } from "react";
import "./App.css";

function App() {
  const [mode, setMode] = useState("upload"); // "upload" | "webcam"

  return (
    <div className="app-container">
      {/* <header className="page-header">
        <div className="title-header">
          <h1>Traffic Sign Detection</h1>
        </div>
        <p>
          Upload an image or video, or use your webcam for live traffic sign
          detection. The app shows the original media and the processed result.
        </p>
      </header> */}

      <div className="mode-tabs">
        <button
          className={mode === "upload" ? "mode-button active" : "mode-button"}
          onClick={() => setMode("upload")}
          aria-label="Upload mode"
        >
          Upload
        </button>
        <button
          className={mode === "webcam" ? "mode-button active" : "mode-button"}
          onClick={() => setMode("webcam")}
          aria-label="Webcam mode"
        >
          Webcam
        </button>
      </div>

      <section className="section-card">
        {mode === "upload" ? <UploadMode /> : <WebcamMode />}
      </section>
    </div>
  );
}

export default App;

//
// ---------------- UPLOAD MODE ----------------
//

function UploadMode() {
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [type, setType] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const fileType = file.type.startsWith("video") ? "video" : "image";

    setType(fileType);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      const contentType = res.headers.get("content-type");

      if (!res.ok) {
        const text = await res.text();
        console.error(text);
        alert("Server error");
        setLoading(false);
        return;
      }

      if (contentType && contentType.includes("application/json")) {
        const err = await res.json();
        alert(err.error || "Processing failed");
        setLoading(false);
        return;
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);

      if (contentType && contentType.startsWith("video")) {
        setType("video");
      } else {
        setType("image");
      }

      setResult(url);
    } catch (err) {
      console.error(err);
      alert("Processing failed");
    }

    setLoading(false);
  };

  return (
    <>
      {/* <h2>Upload Image / Video</h2> */}

      <div className="field-row">
        <label className="file-label">
          Choose File
          <input
            className="file-input"
            type="file"
            accept="image/*,video/*"
            onChange={handleUpload}
          />
        </label>
      </div>

      {loading && <p className="status-text">Processing... videos may take a moment.</p>}

      <div className="preview-grid">
        {preview && (
          <div className="preview-panel">
            <h3>Original</h3>
            {type === "video" ? (
              <video className="preview-media" src={preview} controls />
            ) : (
              <img className="preview-media" src={preview} alt="preview" />
            )}
          </div>
        )}

        {result && (
          <div className="preview-panel">
            <h3>Result</h3>
            {type === "video" ? (
              <video className="preview-media" src={result} controls />
            ) : (
              <img className="preview-media" src={result} alt="result" />
            )}
          </div>
        )}
      </div>
    </>
  );
}

//
// ---------------- WEBCAM MODE ----------------
//

function WebcamMode() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const [socket, setSocket] = useState(null);
  const [output, setOutput] = useState(null);
  const [running, setRunning] = useState(false);

  const streamRef = useRef(null);
  const intervalRef = useRef(null);

  // ---------------- START ----------------
  const start = async () => {
    try {
      // 1. Start webcam
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;

      // 2. Connect WebSocket
      const ws = new WebSocket("ws://localhost:8000/ws");

      ws.onopen = () => console.log("WebSocket connected");

      ws.onmessage = (event) => {
        setOutput(event.data);
      };

      ws.onclose = () => console.log("WebSocket closed");

      setSocket(ws);

      // 3. Start sending frames
      intervalRef.current = setInterval(() => {
        const video = videoRef.current;
        const canvas = canvasRef.current;

        if (!video || !canvas || !ws || ws.readyState !== WebSocket.OPEN) return;
        if (video.readyState !== 4) return;

        const ctx = canvas.getContext("2d");

        const width = 416;
        const height = 416;

        canvas.width = width;
        canvas.height = height;

        ctx.drawImage(video, 0, 0, width, height);

        const dataURL = canvas.toDataURL("image/jpeg", 0.6);
        ws.send(dataURL);
      }, 200);

      setRunning(true);
    } catch (err) {
      console.error("Failed to start webcam:", err);
      alert("Could not start webcam");
    }
  };

  // ---------------- STOP ----------------
  const stop = () => {
    // Stop frame loop
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    // Stop webcam stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    // Close websocket
    if (socket) {
      socket.close();
      setSocket(null);
    }

    setOutput(null);
    setRunning(false);
  };

  // ---------------- TOGGLE ----------------
  const toggle = () => {
    if (running) {
      stop();
    } else {
      start();
    }
  };

  return (
    <>
      {/* <h2>Live Webcam Detection</h2> */}

      <div className="button-row">
        <button className="action-button" onClick={toggle}>
          {running ? "⏹ Stop Webcam" : "▶️ Start Webcam"}
        </button>
        <span className="status-text">
          Status: {running ? "🟢 Running" : "🔴 Stopped"}
        </span>
      </div>

      <div className="preview-grid">
        <div className="preview-panel">
          <h3>Camera</h3>
          <video className="preview-media" ref={videoRef} autoPlay playsInline />
        </div>

        <div className="preview-panel">
          <h3>Processed</h3>
          {output ? (
            <img className="preview-media" src={output} alt="processed" />
          ) : (
            <p className="status-text">No output yet. Start the webcam to view detections.</p>
          )}
        </div>
      </div>

      <canvas ref={canvasRef} style={{ display: "none" }} />
    </>
  );
}