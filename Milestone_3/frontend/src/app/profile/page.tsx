'use client';

import { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
    User, Mail, Phone, MapPin, Calendar, Shield, Heart,
    Edit2, Save, X, Camera, Lock, Activity, LogOut, Settings
} from 'lucide-react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import AnimatedHeart from '@/components/AnimatedHeart';
import FloatingParticles from '@/components/FloatingParticles';
import { getUser, getToken } from '@/services/auth';
import { API_ENDPOINTS } from '@/services/apiClient';

interface UserProfile {
    id: string;
    email: string;
    firstName: string;
    lastName: string;
    phone: string;
    role: 'patient' | 'doctor' | 'admin';
    dateOfBirth: string;
    gender: string;
    address: string;
    city: string;
    country: string;
    bio: string;
    emailVerified: boolean;
    createdAt: string;
    // Doctor specific
    licenseNumber?: string;
    specialization?: string;
    hospital?: string;
}

export default function ProfilePage() {
    const router = useRouter();
    const [isLoading, setIsLoading] = useState(true);
    const [isEditing, setIsEditing] = useState(false);
    const [profile, setProfile] = useState<UserProfile | null>(null);

    const fetchProfile = useCallback(async () => {
        try {
            const token = getToken();
            if (!token) return;

            const res = await fetch(API_ENDPOINTS.auth.profile(), {
                headers: { 'Authorization': `Bearer ${token}` }
            });

            if (res.ok) {
                const data = await res.json();
                setProfile(prev => ({
                    ...prev!,
                    firstName: data.first_name || prev?.firstName || '',
                    lastName: data.last_name || prev?.lastName || '',
                    phone: data.phone || prev?.phone || '',
                    dateOfBirth: data.date_of_birth || prev?.dateOfBirth || '',
                    gender: data.gender || prev?.gender || '',
                    address: data.address || prev?.address || '',
                    city: data.city || prev?.city || '',
                    country: data.country || prev?.country || '',
                    bio: data.bio || prev?.bio || '',
                    emailVerified: data.email_verified ?? prev?.emailVerified ?? false,
                    licenseNumber: data.license_number || prev?.licenseNumber,
                    specialization: data.specialization || prev?.specialization,
                    hospital: data.hospital || prev?.hospital,
                }));
            }
        } catch (error) {
            console.error('Failed to fetch profile:', error);
        }
    }, []);

    useEffect(() => {
        // Get user from localStorage first
        const storedUser = getUser();

        if (!storedUser) {
            router.push('/login');
            return;
        }

        // Set profile from stored user data
        // eslint-disable-next-line react-hooks/set-state-in-effect
        setProfile({
            id: storedUser.id || '1',
            email: storedUser.email || '',
            firstName: storedUser.first_name || storedUser.full_name?.split(' ')[0] || 'User',
            lastName: storedUser.last_name || storedUser.full_name?.split(' ').slice(1).join(' ') || '',
            phone: '',
            role: storedUser.role || 'patient',
            dateOfBirth: '',
            gender: '',
            address: '',
            city: '',
            country: '',
            bio: '',
            emailVerified: storedUser.email_verified || false,
            createdAt: '',
            licenseNumber: storedUser.license_number,
            specialization: storedUser.specialization,
            hospital: storedUser.hospital,
        });

        // Optionally fetch more profile data from backend
        // eslint-disable-next-line react-hooks/set-state-in-effect
        fetchProfile();

        // eslint-disable-next-line react-hooks/set-state-in-effect
        setIsLoading(false);
    }, [router, fetchProfile]);


    const handleSave = async () => {
        if (!profile) return;

        try {
            const token = getToken();
            if (!token) return;

            // Prepare changes - send each field individually like Settings page does
            const updates = [];
            if (profile.phone) updates.push({ field_name: 'phone', new_value: profile.phone });
            if (profile.address) updates.push({ field_name: 'address', new_value: profile.address });
            if (profile.city) updates.push({ field_name: 'city', new_value: profile.city });
            if (profile.country) updates.push({ field_name: 'country', new_value: profile.country });
            if (profile.bio) updates.push({ field_name: 'bio', new_value: profile.bio });

            // Doctor fields
            if (profile.licenseNumber) updates.push({ field_name: 'license_number', new_value: profile.licenseNumber });
            if (profile.specialization) updates.push({ field_name: 'specialization', new_value: profile.specialization });
            if (profile.hospital) updates.push({ field_name: 'hospital', new_value: profile.hospital });

            if (updates.length > 0) {
                let successCount = 0;
                // Send each update individually (matching Settings page behavior)
                for (const update of updates) {
                    const res = await fetch(API_ENDPOINTS.auth.profileChangesSubmit(), {
                        method: 'POST',
                        headers: {
                            'Authorization': `Bearer ${token}`,
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            field_name: update.field_name,
                            new_value: update.new_value,
                            reason: 'Profile update request'
                        })
                    });
                    if (res.ok) successCount++;
                }

                if (successCount > 0) {
                    alert(`${successCount} change(s) submitted for approval!`);
                } else {
                    alert('Failed to submit changes. Please try again.');
                }
            }

            setIsEditing(false);
        } catch (error) {
            alert('Network error. Please check your connection and try again.');
        }
    };

    const handleInputChange = (field: keyof UserProfile, value: string) => {
        if (!profile) return;
        setProfile(prev => prev ? { ...prev, [field]: value } : null);
    };

    if (isLoading || !profile) {
        return (
            <div className="min-h-screen bg-[#0a0a1a] flex items-center justify-center">
                <div className="w-12 h-12 border-4 border-red-500/30 border-t-red-500 rounded-full animate-spin" />
            </div>
        );
    }


    return (
        <div className="min-h-screen bg-[#0a0a1a] relative">
            <div className="fixed inset-0 mesh-bg pointer-events-none" />
            <div className="gradient-orb orb-1" />
            <div className="gradient-orb orb-2" />
            <FloatingParticles />

            {/* Navigation */}
            <nav className="relative z-50 border-b border-white/10">
                <div className="max-w-7xl mx-auto px-6 py-4">
                    <div className="flex items-center justify-between">
                        <Link href="/" className="flex items-center gap-3">
                            <div className="w-8 h-8"><AnimatedHeart className="w-full h-full" /></div>
                            <span className="text-xl font-bold gradient-text">CardioDetect</span>
                        </Link>

                        <div className="flex items-center gap-4">
                            <Link href={profile?.role === 'doctor' ? '/doctor/dashboard' : '/dashboard'} className="text-slate-400 hover:text-white transition-colors">
                                <Activity className="w-5 h-5" />
                            </Link>

                        </div>
                    </div>
                </div>
            </nav>

            {/* Profile Content */}
            <main className="relative z-10 max-w-4xl mx-auto px-6 py-8">
                {/* Header Card */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="glass-card p-8 mb-6"
                >
                    <div className="flex flex-col md:flex-row items-center gap-6">
                        {/* Avatar */}
                        <div className="relative">
                            <div className="w-24 h-24 rounded-full bg-gradient-to-br from-red-500 to-purple-600 flex items-center justify-center text-3xl font-bold text-white">
                                {profile.firstName[0]}{profile.lastName[0]}
                            </div>
                            <button className="absolute bottom-0 right-0 w-8 h-8 rounded-full bg-slate-800 border border-white/20 flex items-center justify-center hover:bg-slate-700 transition-colors">
                                <Camera className="w-4 h-4 text-white" />
                            </button>
                        </div>

                        {/* Info */}
                        <div className="flex-1 text-center md:text-left">
                            <h1 className="text-2xl font-bold text-white mb-1">
                                {profile.role === 'doctor' ? 'Dr. ' : ''}{profile.firstName} {profile.lastName}
                            </h1>
                            <p className="text-slate-400 flex items-center justify-center md:justify-start gap-2">
                                <Mail className="w-4 h-4" />
                                {profile.email}
                                {profile.emailVerified && (
                                    <span className="text-xs px-2 py-0.5 rounded bg-green-500/20 text-green-400">Verified</span>
                                )}
                            </p>
                            <p className="text-sm text-slate-500 mt-1 capitalize flex items-center justify-center md:justify-start gap-2">
                                <Shield className="w-4 h-4" />
                                {profile.role} • Member since {new Date(profile.createdAt).toLocaleDateString()}
                            </p>
                        </div>

                        {/* Edit Button */}
                        <button
                            onClick={() => setIsEditing(!isEditing)}
                            className={`px-4 py-2 rounded-xl flex items-center gap-2 transition-colors ${isEditing
                                ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30'
                                : 'bg-white/10 text-white hover:bg-white/20'
                                }`}
                        >
                            {isEditing ? <X className="w-4 h-4" /> : <Edit2 className="w-4 h-4" />}
                            {isEditing ? 'Cancel' : 'Edit'}
                        </button>
                    </div>
                </motion.div>

                {/* Details Grid */}
                <div className="grid md:grid-cols-2 gap-6">
                    {/* Personal Info */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.1 }}
                        className="glass-card p-6"
                    >
                        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                            <User className="w-5 h-5 text-red-400" />
                            Personal Information
                        </h2>
                        <div className="space-y-4">
                            {[
                                { icon: Phone, label: 'Phone', field: 'phone' as const, value: profile.phone },
                                { icon: Calendar, label: 'Date of Birth', field: 'dateOfBirth' as const, value: profile.dateOfBirth, type: 'date' },
                                { icon: MapPin, label: 'Address', field: 'address' as const, value: profile.address },
                                { icon: MapPin, label: 'City', field: 'city' as const, value: profile.city },
                                { icon: MapPin, label: 'Country', field: 'country' as const, value: profile.country },
                            ].map((item) => (
                                <div key={item.field} className="flex items-center gap-3">
                                    <item.icon className="w-4 h-4 text-slate-500" />
                                    <div className="flex-1">
                                        <label className="text-xs text-slate-500">{item.label}</label>
                                        {isEditing ? (
                                            <input
                                                type={item.type || 'text'}
                                                value={item.value}
                                                onChange={(e) => handleInputChange(item.field, e.target.value)}
                                                className="w-full bg-white/5 border border-white/10 rounded px-2 py-1 text-white text-sm focus:outline-none focus:border-red-500/50"
                                            />
                                        ) : (
                                            <p className="text-white text-sm">{item.value || 'Not set'}</p>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </motion.div>

                    {/* Professional Info (Doctor only) */}
                    {profile.role === 'doctor' && (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.2 }}
                            className="glass-card p-6"
                        >
                            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                                <Heart className="w-5 h-5 text-red-400" />
                                Professional Information
                            </h2>
                            <div className="space-y-4">
                                {[
                                    { label: 'License Number', field: 'licenseNumber' as const, value: profile.licenseNumber },
                                    { label: 'Specialization', field: 'specialization' as const, value: profile.specialization },
                                    { label: 'Hospital', field: 'hospital' as const, value: profile.hospital },
                                ].map((item) => (
                                    <div key={item.field}>
                                        <label className="text-xs text-slate-500">{item.label}</label>
                                        {isEditing ? (
                                            <input
                                                type="text"
                                                value={item.value || ''}
                                                onChange={(e) => handleInputChange(item.field, e.target.value)}
                                                className="w-full bg-white/5 border border-white/10 rounded px-2 py-1 text-white text-sm focus:outline-none focus:border-red-500/50"
                                            />
                                        ) : (
                                            <p className="text-white text-sm">{item.value || 'Not set'}</p>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </motion.div>
                    )}

                    {/* Bio */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.3 }}
                        className="glass-card p-6 md:col-span-2"
                    >
                        <h2 className="text-lg font-semibold text-white mb-4">Bio</h2>
                        {isEditing ? (
                            <textarea
                                value={profile.bio}
                                onChange={(e) => handleInputChange('bio', e.target.value)}
                                rows={3}
                                className="w-full bg-white/5 border border-white/10 rounded px-3 py-2 text-white text-sm focus:outline-none focus:border-red-500/50"
                            />
                        ) : (
                            <p className="text-slate-300 text-sm">{profile.bio || 'No bio provided'}</p>
                        )}
                    </motion.div>

                    {/* Security Section */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.4 }}
                        className="glass-card p-6 md:col-span-2"
                    >
                        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                            <Lock className="w-5 h-5 text-red-400" />
                            Security
                        </h2>
                        <div className="flex flex-wrap gap-4">
                            <Link
                                href="/change-password"
                                className="px-4 py-2 bg-white/10 rounded-xl text-white hover:bg-white/20 transition-colors flex items-center gap-2"
                            >
                                <Lock className="w-4 h-4" />
                                Change Password
                            </Link>
                            <Link
                                href="/settings"
                                className="px-4 py-2 bg-white/10 rounded-xl text-white hover:bg-white/20 transition-colors flex items-center gap-2"
                            >
                                <Settings className="w-4 h-4" />
                                Account Settings
                            </Link>
                        </div>
                    </motion.div>
                </div>




                {/* Save Button */}
                {isEditing && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="mt-6 flex justify-end"
                    >
                        <button
                            onClick={handleSave}
                            className="glow-button px-6 py-3 rounded-xl text-white font-semibold flex items-center gap-2"
                        >
                            <Save className="w-5 h-5" />
                            Save Changes
                        </button>
                    </motion.div>
                )}

                {/* Help Section */}
                <div className="mt-12 text-center">
                    <p className="text-slate-500 text-sm">
                        Need help or found an issue? <Link href="mailto:support@cardiodetect.ai" className="text-blue-400 hover:underline">Click here</Link>
                    </p>
                </div>
            </main>
        </div>
    );
}
