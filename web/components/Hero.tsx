import React from 'react';
import { motion } from 'framer-motion';
import { ArrowDown, Database, ShieldCheck } from 'lucide-react';

export const Hero = () => {
  return (
    <section className="relative h-screen w-full flex flex-col justify-center items-center overflow-hidden border-b border-white/10">
      {/* Background Elements */}
      <div className="absolute inset-0 bg-grid-pattern bg-[size:40px_40px] opacity-10 pointer-events-none"></div>
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-neon blur-[150px] opacity-5 pointer-events-none rounded-full"></div>

      <div className="z-10 text-center max-w-5xl px-4">
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="flex flex-col items-center"
        >
          <div className="flex items-center gap-4 mb-4">
            <span className="w-2 h-2 bg-neon rounded-full animate-pulse"></span>
            <span className="font-mono text-neon text-sm tracking-[0.2em] uppercase">Local LLM Active</span>
          </div>

          <h1 className="font-display text-7xl md:text-9xl font-bold tracking-tighter leading-[0.85] text-white uppercase mb-6 mix-blend-difference">
            Silent<br />
            <span className="text-outline hover:text-white transition-all duration-500">Partner</span>
          </h1>

          <p className="font-mono text-steel max-w-2xl mx-auto text-sm md:text-base leading-relaxed mt-8">
            The local-first document intelligence platform. Ingest sensitive files, contracts, and financial records with zero cloud exposure.
            Talk to your data using on-device AI.
          </p>

          <div className="mt-12 flex flex-col sm:flex-row gap-4">
            <button className="bg-neon/50 text-black/50 font-mono font-bold px-8 py-4 uppercase tracking-wider cursor-not-allowed flex items-center gap-2">
              <Database size={18} />
              Coming Soon
            </button>
            <button className="border border-white/20 text-white font-mono px-8 py-4 uppercase tracking-wider hover:bg-white hover:text-black transition-all flex items-center gap-2">
              <ShieldCheck size={18} />
              Privacy Architecture
            </button>
          </div>
        </motion.div>
      </div>

      <motion.div
        className="absolute bottom-10 left-10 hidden md:block"
        animate={{ y: [0, 10, 0] }}
        transition={{ repeat: Infinity, duration: 2 }}
      >
        <div className="flex flex-col items-center gap-2 text-white/50 font-mono text-xs uppercase writing-mode-vertical">
          <span>Scroll to Analyze</span>
          <ArrowDown size={14} />
        </div>
      </motion.div>

      <div className="absolute bottom-10 right-10 text-right hidden md:block">
        <div className="font-mono text-xs text-white/30">
          OLLAMA COMPATIBLE<br />
          OFFLINE READY
        </div>
      </div>
    </section>
  );
};