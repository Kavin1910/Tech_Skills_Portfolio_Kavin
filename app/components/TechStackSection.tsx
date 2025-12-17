'use client'

import { motion } from 'framer-motion'

interface TechStackSectionProps {
  data: {
    title: string
    items?: string[]
    categories?: Array<{
      name: string
      items?: string[]
      subcategories?: Array<{
        name: string
        items: string[]
      }>
    }>
  }
}

export default function TechStackSection({ data }: TechStackSectionProps) {
  const renderItems = (items: string[]) => (
    <div className="flex flex-wrap gap-2.5 md:gap-3">
      {items.map((item, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0, scale: 0.8, y: 10 }}
          whileInView={{ opacity: 1, scale: 1, y: 0 }}
          viewport={{ once: true, margin: "-50px" }}
          transition={{ delay: index * 0.015, duration: 0.4, type: "spring", stiffness: 100 }}
          whileHover={{ scale: 1.08, y: -2 }}
          className="tech-badge cursor-pointer"
        >
          {item}
        </motion.div>
      ))}
    </div>
  )

  const renderCategories = (categories: Array<{ name: string; items?: string[]; subcategories?: Array<{ name: string; items: string[] }> }>) => (
    <div className="space-y-6 md:space-y-8">
      {categories.map((category, catIndex) => (
        <motion.div
          key={catIndex}
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ delay: catIndex * 0.08, duration: 0.6, type: "spring" }}
          whileHover={{ y: -4 }}
          className="group bg-gray-900/70 backdrop-blur-md rounded-2xl p-6 md:p-8 shadow-md shadow-red-500/10 border border-gray-800/60 hover:shadow-2xl hover:shadow-red-500/20 hover:border-red-500/50 transition-all duration-500 relative overflow-hidden"
        >
          {/* Gradient overlay on hover */}
          <div className="absolute inset-0 bg-gradient-to-br from-red-500/0 to-yellow-500/0 group-hover:from-red-500/10 group-hover:to-yellow-500/10 transition-all duration-500 pointer-events-none"></div>
          
          <div className="relative z-10">
            <h3 className="category-title relative inline-block">
              {category.name}
              <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-gradient-to-r from-red-500 to-yellow-500 group-hover:w-full transition-all duration-500"></span>
            </h3>
            {category.subcategories ? (
              <div className="space-y-5 md:space-y-6 mt-6">
                {category.subcategories.map((subcat, subIndex) => (
                  <motion.div
                    key={subIndex}
                    initial={{ opacity: 0, x: -10 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: subIndex * 0.05, duration: 0.4 }}
                    className="ml-2 md:ml-4"
                  >
                    <h4 className="text-base md:text-lg font-semibold text-gray-200 mb-3 flex items-center">
                      <span className="w-2 h-2 bg-red-500 rounded-full mr-2"></span>
                      {subcat.name}
                    </h4>
                    {renderItems(subcat.items)}
                  </motion.div>
                ))}
              </div>
            ) : category.items ? (
              <div className="mt-4">
                {renderItems(category.items)}
              </div>
            ) : null}
          </div>
        </motion.div>
      ))}
    </div>
  )

  return (
    <motion.section
      initial={{ opacity: 0, y: 40 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-100px" }}
      transition={{ duration: 0.8, type: "spring" }}
      className="group relative bg-gray-900/50 backdrop-blur-lg rounded-3xl p-6 md:p-10 lg:p-12 shadow-xl shadow-red-500/10 border border-gray-800/60 hover:shadow-2xl hover:shadow-red-500/20 hover:border-red-500/50 transition-all duration-500 overflow-hidden"
    >
      {/* Animated background gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-red-500/0 via-red-600/0 to-yellow-500/0 group-hover:from-red-500/10 group-hover:via-red-600/10 group-hover:to-yellow-500/10 transition-all duration-700 pointer-events-none"></div>
      
      <div className="relative z-10">
        <motion.h2
          initial={{ opacity: 0, x: -20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="section-title mb-8 md:mb-10 relative inline-block"
        >
          {data.title}
          <motion.span
            initial={{ width: 0 }}
            whileInView={{ width: "100%" }}
            viewport={{ once: true }}
            transition={{ delay: 0.3, duration: 0.8 }}
            className="absolute -bottom-2 left-0 h-1 bg-gradient-to-r from-red-500 via-red-600 to-yellow-500 rounded-full"
          ></motion.span>
        </motion.h2>
        
        {data.items && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ delay: 0.2, duration: 0.6 }}
          >
            {renderItems(data.items)}
          </motion.div>
        )}

        {data.categories && (
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ delay: 0.2, duration: 0.6 }}
          >
            {renderCategories(data.categories)}
          </motion.div>
        )}
      </div>
    </motion.section>
  )
}

