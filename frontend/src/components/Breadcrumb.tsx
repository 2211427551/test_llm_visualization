import React from 'react';
import { motion } from 'framer-motion';
import { ChevronRight } from 'lucide-react';

interface BreadcrumbProps {
  items: string[];
}

export const Breadcrumb: React.FC<BreadcrumbProps> = ({ items }) => {
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      className="flex items-center gap-2 text-sm text-gray-600"
    >
      {items.map((item, idx) => (
        <React.Fragment key={idx}>
          {idx > 0 && <ChevronRight className="w-4 h-4 text-gray-400" />}
          <span
            className={`${
              idx === items.length - 1
                ? 'font-semibold text-gray-900'
                : 'text-gray-600'
            }`}
          >
            {item}
          </span>
        </React.Fragment>
      ))}
    </motion.div>
  );
};
