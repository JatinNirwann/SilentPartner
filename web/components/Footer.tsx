import React from 'react';

export const Footer = () => {
  return (
    <footer className="bg-black py-20 px-6 border-t border-white/10 font-mono text-sm">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-start md:items-end gap-10">
        <div>
          <div className="w-12 h-12 bg-white text-black flex items-center justify-center font-bold mb-6">
            SP
          </div>
          <p className="text-white/50 max-w-xs">
            Local-first intelligence initiated.
            <br />
            Your data, your machine, your rules.
          </p>
        </div>

        <div className="flex gap-12 text-white/70">
          <div className="flex flex-col gap-4">
            <h4 className="text-white font-bold uppercase tracking-wider">Platform</h4>
            <span className="text-white/30 cursor-not-allowed">Coming Soon</span>
            <a href="#" className="hover:text-neon transition-colors">Documentation</a>
            <a href="#" className="hover:text-neon transition-colors">GitHub</a>
          </div>
          <div className="flex flex-col gap-4">
            <h4 className="text-white font-bold uppercase tracking-wider">Legal</h4>
            <a href="#" className="hover:text-neon transition-colors">Data Privacy</a>
            <a href="#" className="hover:text-neon transition-colors">Terms of Use</a>
            <a href="#" className="hover:text-neon transition-colors">License</a>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto mt-20 pt-8 border-t border-white/10 flex flex-col md:flex-row justify-between text-white/30 text-xs">
        <p>© 2024 SilentPartner Project. All rights reserved.</p>
        <p>Engineered for offline survival.</p>
      </div>
    </footer>
  );
};