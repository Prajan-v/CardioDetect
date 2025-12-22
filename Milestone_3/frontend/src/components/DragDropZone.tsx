'use client';

import { useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileImage, X, Check, AlertCircle } from 'lucide-react';

interface DragDropZoneProps {
    onFileSelect: (file: File) => void;
    acceptedTypes?: string[];
    maxSizeMB?: number;
    className?: string;
}

export default function DragDropZone({
    onFileSelect,
    acceptedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'application/pdf'],
    maxSizeMB = 25,
    className = '',
}: DragDropZoneProps) {
    const [isDragging, setIsDragging] = useState(false);
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const validateFile = useCallback((file: File): string | null => {
        if (!acceptedTypes.includes(file.type)) {
            return `Invalid file type. Accepted: ${acceptedTypes.map(t => t.split('/')[1].toUpperCase()).join(', ')}`;
        }
        if (file.size > maxSizeMB * 1024 * 1024) {
            return `File too large. Maximum size: ${maxSizeMB}MB`;
        }
        return null;
    }, [acceptedTypes, maxSizeMB]);

    const handleFile = useCallback((file: File) => {
        setError(null);
        const validationError = validateFile(file);

        if (validationError) {
            setError(validationError);
            return;
        }

        setSelectedFile(file);
        onFileSelect(file);

        // Generate preview for images
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (e) => setPreview(e.target?.result as string);
            reader.readAsDataURL(file);
        } else {
            setPreview(null);
        }
    }, [validateFile, onFileSelect]);

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(true);
    };

    const handleDragLeave = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    };

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (files && files.length > 0) {
            handleFile(files[0]);
        }
    };

    const clearFile = () => {
        setSelectedFile(null);
        setPreview(null);
        setError(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    return (
        <div className={className}>
            <input
                ref={fileInputRef}
                type="file"
                accept={acceptedTypes.join(',')}
                onChange={handleInputChange}
                className="hidden"
            />

            <motion.div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => !selectedFile && fileInputRef.current?.click()}
                animate={{
                    borderColor: isDragging ? 'rgb(239, 68, 68)' : 'rgba(255, 255, 255, 0.1)',
                    backgroundColor: isDragging ? 'rgba(239, 68, 68, 0.1)' : 'rgba(255, 255, 255, 0.02)',
                }}
                className={`
                    relative border-2 border-dashed rounded-2xl p-8 min-h-[300px]
                    transition-all cursor-pointer
                    ${selectedFile ? 'border-green-500/50 bg-green-500/5' : ''}
                    ${error ? 'border-red-500/50 bg-red-500/5' : ''}
                `}
            >
                <AnimatePresence mode="wait">
                    {selectedFile ? (
                        // File selected state
                        <motion.div
                            key="file-selected"
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            className="flex flex-col items-center"
                        >
                            {/* Preview */}
                            {preview ? (
                                <div className="relative mb-4">
                                    <img
                                        src={preview}
                                        alt="Preview"
                                        className="max-h-48 rounded-xl object-contain border border-white/10"
                                    />
                                    <motion.div
                                        initial={{ scale: 0 }}
                                        animate={{ scale: 1 }}
                                        className="absolute -top-2 -right-2 w-8 h-8 bg-green-500 rounded-full flex items-center justify-center"
                                    >
                                        <Check className="w-5 h-5 text-white" />
                                    </motion.div>
                                </div>
                            ) : (
                                <div className="w-20 h-20 rounded-2xl bg-green-500/20 flex items-center justify-center mb-4">
                                    <FileImage className="w-10 h-10 text-green-400" />
                                </div>
                            )}

                            {/* File info */}
                            <p className="text-white font-medium mb-1 text-center truncate max-w-full">
                                {selectedFile.name}
                            </p>
                            <p className="text-slate-400 text-sm mb-4">
                                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                            </p>

                            {/* Clear button */}
                            <button
                                onClick={(e) => {
                                    e.stopPropagation();
                                    clearFile();
                                }}
                                className="flex items-center gap-2 text-red-400 hover:text-red-300 text-sm transition-colors"
                            >
                                <X className="w-4 h-4" />
                                Remove file
                            </button>
                        </motion.div>
                    ) : error ? (
                        // Error state
                        <motion.div
                            key="error"
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            className="flex flex-col items-center text-center"
                        >
                            <div className="w-16 h-16 rounded-2xl bg-red-500/20 flex items-center justify-center mb-4">
                                <AlertCircle className="w-8 h-8 text-red-400" />
                            </div>
                            <p className="text-red-400 font-medium mb-2">{error}</p>
                            <button
                                onClick={(e) => {
                                    e.stopPropagation();
                                    clearFile();
                                }}
                                className="text-slate-400 hover:text-white text-sm transition-colors"
                            >
                                Try again
                            </button>
                        </motion.div>
                    ) : (
                        // Default state
                        <motion.div
                            key="default"
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            className="flex flex-col items-center text-center"
                        >
                            <motion.div
                                animate={isDragging ? { scale: 1.1, y: -5 } : { scale: 1, y: 0 }}
                                className="w-16 h-16 rounded-2xl bg-red-500/20 flex items-center justify-center mb-4"
                            >
                                <Upload className={`w-8 h-8 ${isDragging ? 'text-red-400' : 'text-red-400/70'}`} />
                            </motion.div>

                            <p className="text-white font-medium mb-2">
                                {isDragging ? 'Drop your file here!' : 'Drag & drop your medical report'}
                            </p>
                            <p className="text-slate-400 text-sm mb-4">
                                or click to browse files
                            </p>

                            <div className="flex flex-wrap justify-center gap-2">
                                {['PNG', 'JPG', 'PDF'].map((type) => (
                                    <span
                                        key={type}
                                        className="px-2 py-1 text-xs bg-white/5 rounded-lg text-slate-400"
                                    >
                                        {type}
                                    </span>
                                ))}
                                <span className="px-2 py-1 text-xs bg-white/5 rounded-lg text-slate-400">
                                    Max {maxSizeMB}MB
                                </span>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Drag overlay pulse effect */}
                {isDragging && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: [0.1, 0.2, 0.1] }}
                        transition={{ duration: 1, repeat: Infinity }}
                        className="absolute inset-0 rounded-2xl bg-red-500/20 pointer-events-none"
                    />
                )}
            </motion.div>
        </div>
    );
}
