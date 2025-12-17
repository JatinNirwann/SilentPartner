import React, { useState, useEffect, useRef } from 'react';
import { motion, useScroll, useTransform, AnimatePresence } from 'framer-motion';
import { Shield, Radio, Zap, Lock, ChevronDown, Smartphone, Globe, EyeOff, Menu, X, ArrowRight, Github } from 'lucide-react';
import { Hero } from './components/Hero';
import { Features } from './components/Features';
import { InterfaceDemo } from './components/InterfaceDemo';
import { Manifesto } from './components/Manifesto';
import { Footer } from './components/Footer';

export default function App() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const { scrollYProgress } = useScroll();
  const progressBarWidth = useTransform(scrollYProgress, [0, 1], ['0%', '100%']);

  return (
    <div className="min-h-screen bg-obsidian font-body selection:bg-neon selection:text-black">
      {/* Progress Bar */}
      <motion.div 
        className="fixed top-0 left-0 h-1 bg-neon z-50"
        style={{ width: progressBarWidth }}
      />

      {/* Navigation */}
      <nav className="fixed top-0 w-full z-40 px-6 py-6 mix-blend-difference text-white">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center gap-2 group cursor-pointer">
            <div className="w-8 h-8 bg-white text-black flex items-center justify-center font-bold font-mono group-hover:bg-neon transition-colors">
              SP
            </div>
            <span className="font-display font-bold text-xl tracking-wider uppercase hidden sm:block">SilentPartner</span>
          </div>

          <div className="hidden md:flex items-center gap-8 font-mono text-sm">
            <a href="#features" className="hover:text-neon transition-colors">MODULES</a>
            <a href="#interface" className="hover:text-neon transition-colors">INTERFACE</a>
            <a href="#manifesto" className="hover:text-neon transition-colors">MANIFESTO</a>
            <a 
              href="https://github.com/JatinNirwann/SilentPartner" 
              target="_blank" 
              rel="noreferrer"
              className="flex items-center gap-2 border border-white px-4 py-2 hover:bg-white hover:text-black transition-all"
            >
              <Github size={16} />
              <span>SOURCE</span>
            </a>
          </div>

          <button 
            className="md:hidden"
            onClick={() => setIsMenuOpen(!isMenuOpen)}
          >
            {isMenuOpen ? <X /> : <Menu />}
          </button>
        </div>
      </nav>

      {/* Mobile Menu */}
      <AnimatePresence>
        {isMenuOpen && (
          <motion.div
            initial={{ y: "-100%" }}
            animate={{ y: 0 }}
            exit={{ y: "-100%" }}
            transition={{ type: "tween", duration: 0.5, ease: "circOut" }}
            className="fixed inset-0 bg-neon z-30 flex flex-col justify-center items-center gap-8 text-black"
          >
            <a onClick={() => setIsMenuOpen(false)} href="#features" className="font-display text-4xl font-bold hover:underline">MODULES</a>
            <a onClick={() => setIsMenuOpen(false)} href="#interface" className="font-display text-4xl font-bold hover:underline">INTERFACE</a>
            <a onClick={() => setIsMenuOpen(false)} href="#manifesto" className="font-display text-4xl font-bold hover:underline">MANIFESTO</a>
          </motion.div>
        )}
      </AnimatePresence>

      <main>
        <Hero />
        <Manifesto />
        <Features />
        <InterfaceDemo />
      </main>

      <Footer />
    </div>
  );
}