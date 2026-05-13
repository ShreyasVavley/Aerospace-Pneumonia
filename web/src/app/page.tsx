"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, X, Activity, FileWarning, CheckCircle, ShieldAlert } from "lucide-react";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [result, setResult] = useState<{
    prediction: string;
    confidence: number;
    heatmap: string;
    probabilities: { Normal: number; Pneumonia: number };
  } | null>(null);

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleFile = (selectedFile: File) => {
    if (selectedFile.type.startsWith("image/")) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setShowHeatmap(false);
    }
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, []);

  const onFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const analyzeScan = async () => {
    if (!file) return;
    setIsAnalyzing(true);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Failed to analyze scan");
      }

      const data = await res.json();
      setResult(data.result);
      setShowHeatmap(true);
    } catch (error) {
      console.error("Inference error:", error);
      alert("Error analyzing the scan. Ensure the FastAPI backend is running.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <main className="min-h-screen relative overflow-hidden flex flex-col items-center justify-center p-6">
      {/* Liquid Background Elements */}
      <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] bg-brand-purple/20 blur-[120px] rounded-full mix-blend-screen pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] bg-brand-emerald/20 blur-[120px] rounded-full mix-blend-screen pointer-events-none" />

      <div className="z-10 w-full max-w-5xl">
        <header className="mb-12 text-center">
          <motion.h1 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-5xl font-bold tracking-tight mb-4 bg-clip-text text-transparent bg-gradient-to-r from-brand-purple to-brand-emerald"
          >
            AeroScan
          </motion.h1>
          <motion.p 
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-foreground/60 text-lg max-w-2xl mx-auto"
          >
            Explainable Pneumonia Detection powered by PyTorch ResNet18 & Grad-CAM.
          </motion.p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Zone */}
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className={`glass-panel rounded-3xl p-8 border-2 transition-all duration-300 flex flex-col items-center justify-center min-h-[400px] relative overflow-hidden ${
              isDragging ? "border-brand-emerald bg-brand-emerald/5" : "border-transparent"
            }`}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
          >
            <input 
              type="file" 
              accept="image/*" 
              onChange={onFileInput}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
              disabled={isAnalyzing}
              aria-label="Upload chest X-ray image for pneumonia analysis"
              title="Upload chest X-ray image"
            />
            
            <AnimatePresence mode="wait">
              {preview ? (
                <motion.div 
                  key="preview"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  className="w-full h-full flex flex-col items-center justify-center relative z-20 pointer-events-none"
                >
                  <div className="relative w-full max-w-sm rounded-2xl overflow-hidden shadow-2xl mb-6 ring-1 ring-white/10 bg-black/40">
                    <img 
                      src={showHeatmap && result?.heatmap ? `data:image/jpeg;base64,${result.heatmap}` : preview} 
                      alt="X-Ray Scan Preview" 
                      className={`w-full h-auto object-cover transition-opacity duration-500 ${showHeatmap ? "opacity-100" : "opacity-80"}`} 
                    />
                    
                    {result && (
                      <div className="absolute top-4 right-4 flex gap-2 pointer-events-auto">
                        <button 
                          onClick={(e) => { e.stopPropagation(); setShowHeatmap(!showHeatmap); }}
                          className={`px-3 py-1.5 rounded-lg text-xs font-bold uppercase tracking-wider transition-all border ${
                            showHeatmap 
                              ? "bg-brand-emerald text-white border-brand-emerald shadow-[0_0_15px_rgba(16,185,129,0.4)]" 
                              : "bg-black/40 text-white/70 border-white/10 hover:bg-black/60"
                          }`}
                        >
                          {showHeatmap ? "Heatmap Active" : "View AI Heatmap"}
                        </button>
                      </div>
                    )}

                    {isAnalyzing && (
                      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center">
                         <div className="flex flex-col items-center gap-4">
                           <Activity className="w-10 h-10 text-brand-emerald animate-pulse" />
                           <p className="text-sm font-medium tracking-widest uppercase text-white">Generating Heatmap</p>
                         </div>
                      </div>
                    )}
                  </div>
                  {!isAnalyzing && (
                    <div className="flex gap-4 pointer-events-auto">
                      <button 
                        onClick={(e) => { e.stopPropagation(); setFile(null); setPreview(null); setResult(null); setShowHeatmap(false); }}
                        className="px-4 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-white transition-colors flex items-center gap-2"
                      >
                        <X className="w-4 h-4" /> Clear
                      </button>
                      <button 
                        onClick={(e) => { e.stopPropagation(); analyzeScan(); }}
                        className="px-6 py-2 rounded-xl bg-gradient-to-r from-brand-purple to-brand-emerald text-white font-semibold shadow-lg hover:opacity-90 transition-opacity flex items-center gap-2"
                      >
                        <Activity className="w-4 h-4" /> Run Inference
                      </button>
                    </div>
                  )}
                </motion.div>
              ) : (
                <motion.div 
                  key="upload"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex flex-col items-center text-center z-0"
                >
                  <div className="w-20 h-20 rounded-full bg-white/5 flex items-center justify-center mb-6 border border-white/10 shadow-inner">
                    <Upload className="w-8 h-8 text-brand-purple" />
                  </div>
                  <h3 className="text-2xl font-semibold mb-2">Upload Chest X-Ray</h3>
                  <p className="text-foreground/50 max-w-xs">
                    Drag and drop your scan here, or click to browse files. Supports JPEG & PNG.
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>

          {/* Results Panel */}
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="glass-panel rounded-3xl p-8 flex flex-col h-full min-h-[400px]"
          >
            <h3 className="text-xl font-semibold mb-6 flex items-center gap-3">
              <Activity className="w-5 h-5 text-brand-emerald" /> 
              Inference Results
            </h3>

            {result ? (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex-1 flex flex-col"
              >
                <div className={`p-6 rounded-2xl mb-8 flex items-start gap-4 border ${
                  result.prediction === "Normal" 
                    ? "bg-brand-emerald/10 border-brand-emerald/20" 
                    : "bg-red-500/10 border-red-500/20"
                }`}>
                  {result.prediction === "Normal" ? (
                    <CheckCircle className="w-8 h-8 text-brand-emerald shrink-0 mt-1" />
                  ) : (
                    <ShieldAlert className="w-8 h-8 text-red-400 shrink-0 mt-1" />
                  )}
                  <div>
                    <p className="text-sm text-foreground/60 uppercase tracking-wider mb-1">Diagnosis</p>
                    <h4 className="text-3xl font-bold">{result.prediction}</h4>
                  </div>
                </div>

                <div className="space-y-6">
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-foreground/70">Confidence Score</span>
                      <span className="font-mono">{(result.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-3 rounded-full bg-white/10 overflow-hidden shadow-inner">
                      <motion.div 
                        initial={{ width: 0 }}
                        animate={{ width: `${result.confidence * 100}%` }}
                        transition={{ duration: 1, ease: "easeOut" }}
                        className="h-full bg-gradient-to-r from-brand-emerald to-brand-purple"
                      />
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4 mt-auto pt-8 border-t border-white/10">
                    <div className="bg-white/5 rounded-xl p-4 border border-white/5 shadow-sm">
                      <p className="text-xs text-foreground/50 mb-1 uppercase tracking-wider">Normal Prob.</p>
                      <p className="font-mono text-lg text-brand-emerald">{(result.probabilities.Normal * 100).toFixed(2)}%</p>
                    </div>
                    <div className="bg-white/5 rounded-xl p-4 border border-white/5 shadow-sm">
                      <p className="text-xs text-foreground/50 mb-1 uppercase tracking-wider">Pneumonia Prob.</p>
                      <p className="font-mono text-lg text-red-400">{(result.probabilities.Pneumonia * 100).toFixed(2)}%</p>
                    </div>
                  </div>

                  <div className="mt-6 p-4 rounded-xl bg-white/5 border border-white/10">
                     <p className="text-xs text-foreground/60 leading-relaxed italic">
                        *AI Heatmap (Grad-CAM) highlights regions of interest that influenced the neural network\u0027s decision.
                     </p>
                  </div>
                </div>
              </motion.div>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center text-foreground/30">
                <FileWarning className="w-16 h-16 mb-4 opacity-50" />
                <p>Awaiting scan data...</p>
              </div>
            )}
          </motion.div>
        </div>
      </div>
    </main>
  );
}
