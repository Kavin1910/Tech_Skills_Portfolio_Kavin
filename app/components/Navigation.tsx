'use client'

import { motion } from 'framer-motion'
import { useState, useEffect } from 'react'

export default function Navigation() {
  const [isScrolled, setIsScrolled] = useState(false)
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50)
    }
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' })
    setIsMobileMenuOpen(false)
  }

  return (
    <motion.nav
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.5 }}
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        isScrolled
          ? 'bg-black/90 backdrop-blur-xl shadow-lg shadow-red-500/10 border-b border-gray-800/50'
          : 'bg-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 md:h-20">
          <motion.button
            onClick={scrollToTop}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="text-xl md:text-2xl font-bold bg-gradient-to-r from-red-500 to-yellow-500 bg-clip-text text-transparent cursor-pointer"
          >
            Kavin N Ranganathan
          </motion.button>

          {/* Desktop Menu */}
          <div className="hidden md:flex items-center space-x-8">
            <a
              href="#my-story"
              className="text-gray-300 hover:text-red-400 transition-colors duration-200 font-medium"
            >
              My Story
            </a>
            <a
              href="#tech-stacks"
              className="text-gray-300 hover:text-red-400 transition-colors duration-200 font-medium"
            >
              Tech Stacks
            </a>
            <a
              href="#contact"
              className="text-gray-300 hover:text-red-400 transition-colors duration-200 font-medium"
            >
              Contact
            </a>
            <motion.button
              onClick={scrollToTop}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="px-6 py-2 bg-gradient-to-r from-red-600 to-red-500 text-white rounded-full font-medium shadow-lg shadow-red-500/30 hover:shadow-xl hover:shadow-red-500/50 transition-all duration-300"
            >
              Back to Top
            </motion.button>
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            className="md:hidden p-2 rounded-lg text-gray-300 hover:bg-gray-800 transition-colors"
            aria-label="Toggle menu"
          >
            <svg
              className="w-6 h-6"
              fill="none"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              {isMobileMenuOpen ? (
                <path d="M6 18L18 6M6 6l12 12" />
              ) : (
                <path d="M4 6h16M4 12h16M4 18h16" />
              )}
            </svg>
          </button>
        </div>

        {/* Mobile Menu */}
        {isMobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden pb-4 space-y-3"
          >
            <a
              href="#my-story"
              onClick={() => setIsMobileMenuOpen(false)}
              className="block text-gray-300 hover:text-red-400 transition-colors duration-200 font-medium py-2"
            >
              My Story
            </a>
            <a
              href="#tech-stacks"
              onClick={() => setIsMobileMenuOpen(false)}
              className="block text-gray-300 hover:text-red-400 transition-colors duration-200 font-medium py-2"
            >
              Tech Stacks
            </a>
            <a
              href="#contact"
              onClick={() => setIsMobileMenuOpen(false)}
              className="block text-gray-300 hover:text-red-400 transition-colors duration-200 font-medium py-2"
            >
              Contact
            </a>
            <button
              onClick={scrollToTop}
              className="w-full px-6 py-2 bg-gradient-to-r from-red-600 to-red-500 text-white rounded-full font-medium shadow-lg shadow-red-500/30"
            >
              Back to Top
            </button>
          </motion.div>
        )}
      </div>
    </motion.nav>
  )
}

