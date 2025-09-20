import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card.jsx';
import { Badge } from './ui/badge.jsx';
import { Button } from './ui/button.jsx';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs.jsx';
import { Progress } from './ui/progress.jsx';
import {
	Users,
	TrendingUp,
	AlertCircle,
	CheckCircle,
	Clock,
	Target,
	BarChart3,
	PieChart,
	Download,
	Eye
} from 'lucide-react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const Overview = () => {
	const [selectedView, setSelectedView] = useState('students');
	const [selectedStudent, setSelectedStudent] = useState(null);

	const statsCardsRef = useRef(null);
	const barChartRef = useRef(null);
	const donutChartRef = useRef(null);
	const timelineRef = useRef(null);
	const performanceCardsRef = useRef(null);

	const statsAnimatedRef = useRef(false);
	const barChartAnimatedRef = useRef(false);
	const donutChartAnimatedRef = useRef(false);
	const timelineAnimatedRef = useRef(false);
	const performanceCardsAnimatedRef = useRef(false);

	// Accessibility: honor reduced motion preference
	const prefersReducedMotion = useRef(false);
	useEffect(() => {
		if (typeof window !== 'undefined' && 'matchMedia' in window) {
			const mql = window.matchMedia('(prefers-reduced-motion: reduce)');
			const setPref = () => { prefersReducedMotion.current = mql.matches; };
			setPref();
			mql.addEventListener('change', setPref);
			return () => mql.removeEventListener('change', setPref);
		}
	}, []);

	const isInViewport = (el) => {
		if (!el) return false;
		const rect = el.getBoundingClientRect();
		const vh = window.innerHeight || document.documentElement.clientHeight;
		const vw = window.innerWidth || document.documentElement.clientWidth;
		return rect.bottom >= 0 && rect.right >= 0 && rect.top <= vh && rect.left <= vw;
	};

	useEffect(() => {
		if (statsCardsRef.current && !statsAnimatedRef.current) {
			// If reduced motion is preferred, set final states instantly
			if (prefersReducedMotion.current) {
				const cards = statsCardsRef.current.querySelectorAll('.stat-card');
				gsap.set(cards, { opacity: 1, y: 0, scale: 1 });
				const counters = statsCardsRef.current.querySelectorAll('.counter-number');
				counters.forEach((counter) => {
					const target = counter.textContent || '';
					counter.textContent = target;
				});
				statsAnimatedRef.current = true;
				return;
			}

			const cards = statsCardsRef.current.querySelectorAll('.stat-card');
			gsap.fromTo(
				cards,
				{ opacity: 0, y: 50, scale: 0.9 },
				{
					opacity: 1,
					y: 0,
					scale: 1,
					duration: 0.8,
					stagger: 0.15,
					ease: 'back.out(1.7)',
					onStart: () => { statsAnimatedRef.current = true; },
					scrollTrigger: {
						trigger: statsCardsRef.current,
						start: 'top 80%',
						once: true
					}
				}
			);

			const counters = statsCardsRef.current.querySelectorAll('.counter-number');
			counters.forEach((counter) => {
				const target = parseInt(counter.textContent || '0');
				gsap.to({ value: 0 }, {
					value: target,
					duration: 2,
					ease: 'power2.out',
					onUpdate: function () {
						counter.textContent = Math.round(this.targets()[0].value).toString();
					},
					scrollTrigger: {
						trigger: counter,
						start: 'top 80%',
						once: true
					}
				});
			});
		}
	}, []);

	useEffect(() => {
		// Reduced motion: immediately set final states and skip registering animations
		if (prefersReducedMotion.current) {
			// Bar chart finals
			if (barChartRef.current && !barChartAnimatedRef.current) {
				const bars = barChartRef.current.querySelectorAll('.chart-bar');
				gsap.set(bars, { scaleY: 1, transformOrigin: 'bottom' });
				const labels = barChartRef.current.querySelectorAll('.bar-label');
				gsap.set(labels, { opacity: 1, y: 0 });
				barChartAnimatedRef.current = true;
			}

			// Donut chart finals
			if (donutChartRef.current && !donutChartAnimatedRef.current) {
				const segments = donutChartRef.current.querySelectorAll('.donut-segment');
				segments.forEach((circle) => {
					const dash = circle.getAttribute('data-dash');
					if (dash) circle.setAttribute('stroke-dasharray', dash);
				});
				const centerText = donutChartRef.current.querySelector('.center-text');
				if (centerText) gsap.set(centerText, { scale: 1, opacity: 1 });
				donutChartAnimatedRef.current = true;
			}

			// Timeline finals
			if (timelineRef.current && !timelineAnimatedRef.current) {
				const timelineLine = timelineRef.current.querySelector('.timeline-line');
				if (timelineLine) gsap.set(timelineLine, { scaleY: 1, transformOrigin: 'top' });
				const points = timelineRef.current.querySelectorAll('.timeline-point');
				gsap.set(points, { scale: 1, opacity: 1 });
				const progressLine = timelineRef.current.querySelector('.progress-line');
				if (progressLine && typeof progressLine.getTotalLength === 'function') {
					const pathLength = progressLine.getTotalLength();
					progressLine.setAttribute('stroke-dasharray', `${pathLength} ${pathLength}`);
				}
				timelineAnimatedRef.current = true;
			}

			// Performance cards finals
			if (performanceCardsRef.current && !performanceCardsAnimatedRef.current) {
				const cards = performanceCardsRef.current.querySelectorAll('.performance-card');
				gsap.set(cards, { opacity: 1, y: 0, rotationX: 0 });
				const fills = performanceCardsRef.current.querySelectorAll('.meter-fill');
				fills.forEach((fill) => {
					const w = fill.getAttribute('data-width') || '0%';
					fill.style.width = w;
				});
				performanceCardsAnimatedRef.current = true;
			}

			ScrollTrigger.refresh();
			return;
		}

		if (selectedView === 'analytics') {
			if (barChartRef.current && !barChartAnimatedRef.current && isInViewport(barChartRef.current)) {
				const bars = barChartRef.current.querySelectorAll('.chart-bar');
				gsap.fromTo(bars, { scaleY: 0, transformOrigin: 'bottom' }, { scaleY: 1, duration: 1.0, stagger: 0.15, ease: 'power3.out' });
				const labels = barChartRef.current.querySelectorAll('.bar-label');
				gsap.fromTo(labels, { opacity: 0, y: -20 }, { opacity: 1, y: 0, duration: 0.5, stagger: 0.15, delay: 0.6, ease: 'power2.out' });
				// Smoothly count bar label numbers from 0 to their target values
				labels.forEach((label) => {
					const target = parseFloat((label.textContent || '0').replace(/[^0-9.]/g, '')) || 0;
					gsap.fromTo(
						{ val: 0 },
						{ val: target, duration: 0.8, delay: 0.6, ease: 'power2.out', onUpdate() { label.textContent = this.targets()[0].val.toFixed(1); } }
					);
				});
				barChartAnimatedRef.current = true;
			}
			if (donutChartRef.current && !donutChartAnimatedRef.current && isInViewport(donutChartRef.current)) {
				const segments = donutChartRef.current.querySelectorAll('.donut-segment');
				segments.forEach((segment, index) => {
					const circle = segment;
					const circumference = 2 * Math.PI * parseFloat(circle.getAttribute('r') || '0');
					gsap.fromTo(circle, { strokeDasharray: `0 ${circumference}` }, {
						strokeDasharray: circle.getAttribute('data-dash') || `${circumference} ${circumference}`,
						duration: 1.5,
						delay: index * 0.3,
						ease: 'power2.out'
					});
				});
				const centerText = donutChartRef.current.querySelector('.center-text');
				if (centerText) {
					gsap.fromTo(centerText, { scale: 0, opacity: 0 }, { scale: 1, opacity: 1, duration: 0.8, delay: 1.5, ease: 'back.out(1.7)' });
				}
				donutChartAnimatedRef.current = true;
			}
			if (timelineRef.current && !timelineAnimatedRef.current && isInViewport(timelineRef.current)) {
				const timelineLine = timelineRef.current.querySelector('.timeline-line');
				if (timelineLine) {
					gsap.fromTo(timelineLine, { scaleY: 0, transformOrigin: 'top' }, { scaleY: 1, duration: 2, ease: 'power2.out' });
				}
				const points = timelineRef.current.querySelectorAll('.timeline-point');
				gsap.fromTo(points, { scale: 0, opacity: 0 }, { scale: 1, opacity: 1, duration: 0.6, stagger: 0.3, delay: 0.5, ease: 'back.out(1.7)' });
				const progressLine = timelineRef.current.querySelector('.progress-line');
				if (progressLine && typeof progressLine.getTotalLength === 'function') {
					const pathLength = progressLine.getTotalLength();
					gsap.fromTo(progressLine, { strokeDasharray: `0 ${pathLength}` }, { strokeDasharray: `${pathLength} ${pathLength}`, duration: 3, delay: 1, ease: 'power2.out' });
				}
				timelineAnimatedRef.current = true;
			}
			if (performanceCardsRef.current && !performanceCardsAnimatedRef.current && isInViewport(performanceCardsRef.current)) {
				const cards = performanceCardsRef.current.querySelectorAll('.performance-card');
				gsap.fromTo(cards, { opacity: 0, y: 30, rotationX: -15 }, { opacity: 1, y: 0, rotationX: 0, duration: 0.7, stagger: 0.12, ease: 'power2.out' });
				// Animate meter bar widths smoothly on first reveal
				const fills = performanceCardsRef.current.querySelectorAll('.meter-fill');
				fills.forEach((fill) => {
					const targetWidth = fill.getAttribute('data-width') || '0%';
					gsap.to(fill, { width: targetWidth, duration: 0.9, ease: 'power2.out', delay: 0.15 });
				});
				performanceCardsAnimatedRef.current = true;
			}
		}

		if (barChartRef.current && !barChartAnimatedRef.current) {
			const bars = barChartRef.current.querySelectorAll('.chart-bar');
			gsap.fromTo(bars, { scaleY: 0, transformOrigin: 'bottom' }, {
				scaleY: 1,
				duration: 1.0,
				stagger: 0.15,
				ease: 'power3.out',
				scrollTrigger: {
					trigger: barChartRef.current,
					start: 'top 80%',
					once: true,
					onEnter: () => { barChartAnimatedRef.current = true; }
				}
			});

			const labels = barChartRef.current.querySelectorAll('.bar-label');
			gsap.fromTo(labels, { opacity: 0, y: -20 }, {
				opacity: 1,
				y: 0,
				duration: 0.5,
				stagger: 0.15,
				delay: 0.6,
				ease: 'power2.out',
				scrollTrigger: {
					trigger: barChartRef.current,
					start: 'top 80%',
					once: true
				}
			});
			// Counter animation for labels on scroll reveal
			labels.forEach((label) => {
				const target = parseFloat((label.textContent || '0').replace(/[^0-9.]/g, '')) || 0;
				gsap.fromTo(
					{ val: 0 },
					{
						val: target,
						duration: 0.8,
						delay: 0.6,
						ease: 'power2.out',
						scrollTrigger: {
							trigger: barChartRef.current,
							start: 'top 80%',
							once: true
						},
						onUpdate() { label.textContent = this.targets()[0].val.toFixed(1); }
					}
				);
			});
		}

		if (donutChartRef.current && !donutChartAnimatedRef.current) {
			const segments = donutChartRef.current.querySelectorAll('.donut-segment');
			segments.forEach((segment, index) => {
				const circle = segment;
				const circumference = 2 * Math.PI * parseFloat(circle.getAttribute('r') || '0');
				gsap.fromTo(circle, { strokeDasharray: `0 ${circumference}` }, {
					strokeDasharray: circle.getAttribute('data-dash') || `${circumference} ${circumference}`,
					duration: 1.5,
					delay: index * 0.3,
					ease: 'power2.out',
					scrollTrigger: {
						trigger: donutChartRef.current,
						start: 'top 80%',
						once: true,
						onEnter: index === 0 ? () => { donutChartAnimatedRef.current = true; } : undefined
					}
				});
			});

			const centerText = donutChartRef.current.querySelector('.center-text');
			if (centerText) {
				gsap.fromTo(centerText, { scale: 0, opacity: 0 }, {
					scale: 1,
					opacity: 1,
					duration: 0.8,
					delay: 1.5,
					ease: 'back.out(1.7)',
					scrollTrigger: {
						trigger: donutChartRef.current,
						start: 'top 80%',
						once: true
					}
				});
			}
		}

		if (timelineRef.current && !timelineAnimatedRef.current) {
			const timelineLine = timelineRef.current.querySelector('.timeline-line');
			if (timelineLine) {
				gsap.fromTo(timelineLine, { scaleY: 0, transformOrigin: 'top' }, {
					scaleY: 1,
					duration: 2,
					ease: 'power2.out',
					scrollTrigger: {
						trigger: timelineRef.current,
						start: 'top 80%',
						once: true,
						onEnter: () => { timelineAnimatedRef.current = true; }
					}
				});
			}

			const points = timelineRef.current.querySelectorAll('.timeline-point');
			gsap.fromTo(points, { scale: 0, opacity: 0 }, {
				scale: 1,
				opacity: 1,
				duration: 0.6,
				stagger: 0.3,
				delay: 0.5,
				ease: 'back.out(1.7)',
				scrollTrigger: {
					trigger: timelineRef.current,
					start: 'top 80%',
					once: true
				}
			});

			const progressLine = timelineRef.current.querySelector('.progress-line');
			if (progressLine) {
				const pathLength = (progressLine.getTotalLength && progressLine.getTotalLength()) || 1000;
				gsap.fromTo(progressLine, { strokeDasharray: `0 ${pathLength}` }, {
					strokeDasharray: `${pathLength} ${pathLength}`,
					duration: 3,
					delay: 1,
					ease: 'power2.out',
					scrollTrigger: {
						trigger: timelineRef.current,
						start: 'top 80%',
						once: true
					}
				});
			}
		}

		if (performanceCardsRef.current && !performanceCardsAnimatedRef.current) {
			const cards = performanceCardsRef.current.querySelectorAll('.performance-card');
			gsap.fromTo(cards, { opacity: 0, y: 30, rotationX: -15 }, {
				opacity: 1,
				y: 0,
				rotationX: 0,
				duration: 0.7,
				stagger: 0.12,
				ease: 'power2.out',
				scrollTrigger: {
					trigger: performanceCardsRef.current,
					start: 'top 80%',
					once: true,
					onEnter: () => { performanceCardsAnimatedRef.current = true; }
				}
			});

			// Animate the width of meter bars on scroll reveal
			const fills = performanceCardsRef.current.querySelectorAll('.meter-fill');
			fills.forEach((fill) => {
				const targetWidth = fill.getAttribute('data-width') || '0%';
				gsap.to(fill, {
					width: targetWidth,
					duration: 0.9,
					ease: 'power2.out',
					scrollTrigger: {
						trigger: performanceCardsRef.current,
						start: 'top 80%',
						once: true
					}
				});
			});
		}

		ScrollTrigger.refresh();
		return () => {
			ScrollTrigger.getAll().forEach((trigger) => {
				if (trigger.trigger !== statsCardsRef.current) trigger.kill();
			});
		};
	}, [selectedView]);

	useEffect(() => {
		return () => {
			ScrollTrigger.getAll().forEach((t) => t.kill());
		};
	}, []);

	const overviewStats = {
		totalStudents: 150,
		totalSubjects: 8,
		completedEvaluations: 1200,
		pendingEvaluations: 45,
		averageClassScore: 78.5,
		passingRate: 85.2,
		topPerformers: 25,
		needsAttention: 18
	};

	const studentSummaries = [
		{ id: '1', name: 'John Doe', rollNumber: 'R001', totalSubjects: 8, completedSubjects: 8, averageScore: 92.5, highestScore: 98, lowestScore: 85, status: 'excellent', lastActivity: '2024-01-15' },
		{ id: '2', name: 'Jane Smith', rollNumber: 'R002', totalSubjects: 8, completedSubjects: 7, averageScore: 78.3, highestScore: 89, lowestScore: 65, status: 'good', lastActivity: '2024-01-14' },
		{ id: '3', name: 'Mike Johnson', rollNumber: 'R003', totalSubjects: 8, completedSubjects: 6, averageScore: 58.7, highestScore: 72, lowestScore: 42, status: 'needs-improvement', lastActivity: '2024-01-12' }
	];

	const subjectSummaries = [
		{ subject: 'Mathematics', totalStudents: 150, completedStudents: 148, averageScore: 74.2, highestScore: 98, lowestScore: 32, passingRate: 82.4, difficultyLevel: 'hard' },
		{ subject: 'Physics', totalStudents: 150, completedStudents: 145, averageScore: 79.1, highestScore: 96, lowestScore: 45, passingRate: 87.6, difficultyLevel: 'medium' },
		{ subject: 'Chemistry', totalStudents: 150, completedStudents: 142, averageScore: 81.3, highestScore: 99, lowestScore: 52, passingRate: 89.4, difficultyLevel: 'medium' }
	];

	const getStatusColor = (status) => {
		switch (status) {
			case 'excellent': return 'bg-green-100 text-green-800 border-green-200';
			case 'good': return 'bg-blue-100 text-blue-800 border-blue-200';
			case 'average': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
			case 'needs-improvement': return 'bg-red-100 text-red-800 border-red-200';
			default: return 'bg-gray-100 text-gray-800 border-gray-200';
		}
	};

	const getDifficultyColor = (difficulty) => {
		switch (difficulty) {
			case 'easy': return 'bg-green-100 text-green-800';
			case 'medium': return 'bg-yellow-100 text-yellow-800';
			case 'hard': return 'bg-red-100 text-red-800';
			default: return 'bg-gray-100 text-gray-800';
		}
	};

	return (
		<div className="space-y-6 no-select">
			<div className="flex justify-between items-center">
				<div>
					<h2 className="text-3xl font-bold text-foreground no-select">Evaluation Overview</h2>
					<p className="text-muted-foreground no-select">Comprehensive analysis of student performance and evaluation metrics</p>
				</div>
				{/* Removed Filter and Export buttons per request */}
			</div>

			<div ref={statsCardsRef} className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
				<Card className="no-select stat-card">
					<CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
						<CardTitle className="text-sm font-medium no-select">Total Students</CardTitle>
						<Users className="h-4 w-4 text-muted-foreground" />
					</CardHeader>
					<CardContent>
						<div className="text-2xl font-bold text-foreground no-select counter-number">{overviewStats.totalStudents}</div>
						<p className="text-xs text-muted-foreground no-select">Across {overviewStats.totalSubjects} subjects</p>
					</CardContent>
				</Card>

				<Card className="no-select stat-card">
					<CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
						<CardTitle className="text-sm font-medium no-select">Average Score</CardTitle>
						<Target className="h-4 w-4 text-muted-foreground" />
					</CardHeader>
					<CardContent>
						<div className="text-2xl font-bold text-foreground no-select counter-number">{overviewStats.averageClassScore}</div>
						<p className="text-xs text-green-600 no-select">
							<TrendingUp className="h-3 w-3 inline mr-1" />
							+2.3% from last batch
						</p>
					</CardContent>
				</Card>

				<Card className="no-select stat-card">
					<CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
						<CardTitle className="text-sm font-medium no-select">Passing Rate</CardTitle>
						<CheckCircle className="h-4 w-4 text-muted-foreground" />
					</CardHeader>
					<CardContent>
						<div className="text-2xl font-bold text-foreground no-select counter-number">{overviewStats.passingRate}</div>
						<p className="text-xs text-green-600 no-select">
							<TrendingUp className="h-3 w-3 inline mr-1" />
							{overviewStats.totalStudents - Math.floor(overviewStats.totalStudents * overviewStats.passingRate / 100)} students passed
						</p>
					</CardContent>
				</Card>

				<Card className="no-select stat-card">
					<CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
						<CardTitle className="text-sm font-medium no-select">Needs Attention</CardTitle>
						<AlertCircle className="h-4 w-4 text-muted-foreground" />
					</CardHeader>
					<CardContent>
						<div className="text-2xl font-bold text-red-600 no-select counter-number">{overviewStats.needsAttention}</div>
						<p className="text-xs text-muted-foreground no-select">Students requiring support</p>
					</CardContent>
				</Card>
			</div>

			<Tabs value={selectedView} onValueChange={(value) => setSelectedView(value)} className="no-select">
				<TabsList className="grid w-full grid-cols-3 no-select">
					<TabsTrigger value="students" className="no-select">Student Performance</TabsTrigger>
					<TabsTrigger value="subjects" className="no-select">Subject Analysis</TabsTrigger>
					<TabsTrigger value="analytics" className="no-select">Advanced Analytics</TabsTrigger>
				</TabsList>

				<TabsContent value="students" className="space-y-4">
					<div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
						<div className="lg:col-span-2">
							<Card className="no-select">
								<CardHeader>
									<CardTitle className="no-select">Student Performance Summary</CardTitle>
									<CardDescription className="no-select">Click on a student to view detailed performance metrics</CardDescription>
								</CardHeader>
								<CardContent>
									<div className="space-y-3">
										{studentSummaries.map((student) => (
											<div key={student.id} className={`p-4 border rounded-lg cursor-pointer no-select ${selectedStudent === student.id ? 'border-primary bg-primary/5' : ''}`} onClick={() => setSelectedStudent(student.id)}>
												<div className="flex justify-between items-start">
													<div className="flex-1">
														<div className="flex items-center gap-3">
															<h4 className="font-semibold text-foreground no-select">{student.name}</h4>
															<Badge variant="outline" className="no-select">{student.rollNumber}</Badge>
															<Badge className={getStatusColor(student.status)} variant="outline">{student.status.replace('-', ' ')}</Badge>
														</div>
														<div className="mt-2 flex items-center gap-4 text-sm text-muted-foreground">
															<span className="no-select">Progress: {student.completedSubjects}/{student.totalSubjects}</span>
															<span className="no-select">Avg: {student.averageScore}%</span>
															<span className="no-select">Range: {student.lowestScore}% - {student.highestScore}%</span>
														</div>
													</div>
													<Button variant="ghost" size="sm" className="no-select button-hover-pulse button-icon-rotate">
														<Eye className="h-4 w-4" />
													</Button>
												</div>
												<div className="mt-3">
													<div className="flex justify-between text-xs text-muted-foreground mb-1">
														<span className="no-select">Completion Progress</span>
														<span className="no-select">{Math.round((student.completedSubjects / student.totalSubjects) * 100)}%</span>
													</div>
													<Progress value={(student.completedSubjects / student.totalSubjects) * 100} className="h-2" />
												</div>
											</div>
										))}
									</div>
								</CardContent>
							</Card>
						</div>

						<div>
							<Card className="no-select">
								<CardHeader>
									<CardTitle className="no-select">Performance Insights</CardTitle>
								</CardHeader>
								<CardContent>
									{selectedStudent ? (
										<div className="space-y-4">
											{(() => {
												const student = studentSummaries.find((s) => s.id === selectedStudent);
												if (!student) return null;
												return (
													<>
														<div>
															<h4 className="font-semibold text-foreground no-select">{student.name}</h4>
															<p className="text-sm text-muted-foreground no-select">{student.rollNumber}</p>
														</div>
														<div className="space-y-3">
															<div className="flex justify-between"><span className="text-sm text-muted-foreground no-select">Average Score</span><span className="font-medium text-foreground no-select">{student.averageScore}%</span></div>
															<div className="flex justify-between"><span className="text-sm text-muted-foreground no-select">Highest Score</span><span className="font-medium text-green-600 no-select">{student.highestScore}%</span></div>
															<div className="flex justify-between"><span className="text-sm text-muted-foreground no-select">Lowest Score</span><span className="font-medium text-red-600 no-select">{student.lowestScore}%</span></div>
															<div className="flex justify-between"><span className="text-sm text-muted-foreground no-select">Completion</span><span className="font-medium text-foreground no-select">{student.completedSubjects}/{student.totalSubjects}</span></div>
														</div>
														<div className="pt-4 border-t">
															<h5 className="font-medium mb-2 text-foreground no-select">Recommendations</h5>
															<div className="text-sm text-muted-foreground space-y-1">
																{student.status === 'excellent' && <p className="no-select">🌟 Excellent performance! Consider advanced challenges.</p>}
																{student.status === 'good' && <p className="no-select">👍 Good progress. Focus on consistency across subjects.</p>}
																{student.status === 'needs-improvement' && <p className="no-select">📚 Needs additional support. Schedule one-on-one sessions.</p>}
															</div>
														</div>
													</>
												);
											})()}
										</div>
									) : (
										<p className="text-sm text-muted-foreground no-select">Select a student from the list to view detailed performance insights.</p>
									)}
								</CardContent>
							</Card>
						</div>
					</div>
				</TabsContent>

				<TabsContent value="subjects" className="space-y-4">
					<div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
						{subjectSummaries.map((subject) => (
							<Card key={subject.subject} className="no-select">
								<CardHeader>
									<div className="flex justify-between items-start">
										<div>
											<CardTitle className="no-select">{subject.subject}</CardTitle>
											<CardDescription className="no-select">{subject.completedStudents}/{subject.totalStudents} students completed</CardDescription>
										</div>
										<Badge className={getDifficultyColor(subject.difficultyLevel)} variant="secondary">{subject.difficultyLevel}</Badge>
									</div>
								</CardHeader>
								<CardContent>
									<div className="space-y-4">
										<div className="grid grid-cols-3 gap-4 text-center">
											<div><div className="text-lg font-bold text-foreground no-select">{subject.averageScore}%</div><div className="text-xs text-muted-foreground no-select">Average</div></div>
											<div><div className="text-lg font-bold text-green-600 no-select">{subject.highestScore}%</div><div className="text-xs text-muted-foreground no-select">Highest</div></div>
											<div><div className="text-lg font-bold text-red-600 no-select">{subject.lowestScore}%</div><div className="text-xs text-muted-foreground no-select">Lowest</div></div>
										</div>
										<div>
											<div className="flex justify-between text-sm mb-2"><span className="text-muted-foreground no-select">Passing Rate</span><span className="font-medium text-foreground no-select">{subject.passingRate}%</span></div>
											<Progress value={subject.passingRate} className="h-2" />
										</div>
										<div>
											<div className="flex justify-between text-sm mb-2"><span className="text-muted-foreground no-select">Completion Rate</span><span className="font-medium text-foreground no-select">{Math.round((subject.completedStudents / subject.totalStudents) * 100)}%</span></div>
											<Progress value={(subject.completedStudents / subject.totalStudents) * 100} className="h-2" />
										</div>
									</div>
								</CardContent>
							</Card>
						))}
					</div>
				</TabsContent>

				<TabsContent value="analytics" className="space-y-4">
					<div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
						<Card className="no-select">
							<CardHeader className="flex flex-row items-center justify-between pb-2">
								<CardTitle className="text-lg font-semibold no-select">Subject-wise Performance</CardTitle>
								{/* Removed Export button */}
							</CardHeader>
							<CardContent>
								<div className="space-y-4">
									<div ref={barChartRef} className="relative h-64 w-full">
										<div className="flex items-end justify-between h-full pb-8 gap-2">
											<div className="flex flex-col items-center h-full justify-end">
												<div className="bg-blue-500 rounded-t-sm mb-2 relative group chart-bar" style={{ height: '85%', width: '40px' }}>
													<div className="absolute -top-6 left-1/2 transform -translate-x-1/2 text-xs font-medium text-foreground no-select bar-label">76.2</div>
												</div>
												<span className="text-xs text-muted-foreground no-select transform -rotate-45 origin-bottom">Data Analysis</span>
											</div>
											<div className="flex flex-col items-center h-full justify-end">
												<div className="bg-blue-500 rounded-t-sm mb-2 relative group chart-bar" style={{ height: '70%', width: '40px' }}>
													<div className="absolute -top-6 left-1/2 transform -translate-x-1/2 text-xs font-medium text-foreground no-select bar-label">62.8</div>
												</div>
												<span className="text-xs text-muted-foreground no-select transform -rotate-45 origin-bottom">Machine Learning</span>
											</div>
											<div className="flex flex-col items-center h-full justify-end">
												<div className="bg-blue-600 rounded-t-sm mb-2 relative group chart-bar" style={{ height: '75%', width: '40px' }}>
													<div className="absolute -top-6 left-1/2 transform -translate-x-1/2 text-xs font-medium text-foreground no-select bar-label">69.4</div>
												</div>
												<span className="text-xs text-muted-foreground no-select transform -rotate-45 origin-bottom">Programming</span>
											</div>
											<div className="flex flex-col items-center h-full justify-end">
												<div className="bg-blue-600 rounded-t-sm mb-2 relative group chart-bar" style={{ height: '68%', width: '40px' }}>
													<div className="absolute -top-6 left-1/2 transform -translate-x-1/2 text-xs font-medium text-foreground no-select bar-label">65.7</div>
												</div>
												<span className="text-xs text-muted-foreground no-select transform -rotate-45 origin-bottom">Statistics</span>
											</div>
											<div className="flex flex-col items-center h-full justify-end">
												<div className="bg-blue-600 rounded-t-sm mb-2 relative group chart-bar" style={{ height: '72%', width: '40px' }}>
													<div className="absolute -top-6 left-1/2 transform -translate-x-1/2 text-xs font-medium text-foreground no-select bar-label">68.5</div>
												</div>
												<span className="text-xs text-muted-foreground no-select transform -rotate-45 origin-bottom">Web Development</span>
											</div>
										</div>
										<div className="absolute left-0 top-0 h-full flex flex-col justify-between text-xs text-muted-foreground py-2">
											<span className="no-select">80</span>
											<span className="no-select">60</span>
											<span className="no-select">40</span>
											<span className="no-select">20</span>
											<span className="no-select">0</span>
										</div>
									</div>
								</div>
							</CardContent>
						</Card>

						<Card className="no-select">
							<CardHeader>
								<CardTitle className="text-lg font-semibold no-select">Score Distribution</CardTitle>
							</CardHeader>
							<CardContent>
								<div className="space-y-6">
									<div ref={donutChartRef} className="relative flex items-center justify-center">
										<div className="relative w-48 h-48">
											<svg className="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
												<circle cx="18" cy="18" r="15.5" fill="transparent" stroke="#10b981" strokeWidth="3" strokeDasharray="7.5 92.5" strokeDashoffset="0" className="donut-segment" data-dash="7.5 92.5" />
												<circle cx="18" cy="18" r="15.5" fill="transparent" stroke="#06b6d4" strokeWidth="3" strokeDasharray="21.2 78.8" strokeDashoffset="-7.5" className="donut-segment" data-dash="21.2 78.8" />
												<circle cx="18" cy="18" r="15.5" fill="transparent" stroke="#3b82f6" strokeWidth="3" strokeDasharray="37.1 62.9" strokeDashoffset="-28.7" className="donut-segment" data-dash="37.1 62.9" />
												<circle cx="18" cy="18" r="15.5" fill="transparent" stroke="#f59e0b" strokeWidth="3" strokeDasharray="34.2 65.8" strokeDashoffset="-65.8" className="donut-segment" data-dash="34.2 65.8" />
											</svg>
											<div className="absolute inset-0 flex flex-col items-center justify-center center-text">
												<div className="text-xs text-muted-foreground no-select">Count: 834 students</div>
											</div>
										</div>
									</div>

									<div className="space-y-3">
										<div className="flex justify-between items-center">
											<div className="flex items-center gap-2"><div className="w-3 h-3 bg-blue-500 rounded-full"></div><span className="text-sm text-muted-foreground no-select">90-100%</span></div>
											<div className="text-right"><div className="text-sm font-medium text-foreground no-select">245 students</div><div className="text-xs text-muted-foreground no-select">7.5%</div></div>
										</div>
										<div className="flex justify-between items-center">
											<div className="flex items-center gap-2"><div className="w-3 h-3 bg-cyan-500 rounded-full"></div><span className="text-sm text-muted-foreground no-select">80-89%</span></div>
											<div className="text-right"><div className="text-sm font-medium text-foreground no-select">687 students</div><div className="text-xs text-muted-foreground no-select">21.2%</div></div>
										</div>
										<div className="flex justify-between items-center">
											<div className="flex items-center gap-2"><div className="w-3 h-3 bg-green-500 rounded-full"></div><span className="text-sm text-muted-foreground no-select">70-79%</span></div>
											<div className="text-right"><div className="text-sm font-medium text-foreground no-select">1205 students</div><div className="text-xs text-muted-foreground no-select">37.1%</div></div>
										</div>
									</div>
								</div>
							</CardContent>
						</Card>
					</div>

					<div ref={performanceCardsRef} className="grid grid-cols-1 lg:grid-cols-2 gap-6">
						<Card className="no-select performance-card">
							<CardHeader>
								<CardTitle className="flex items-center gap-2 no-select"><BarChart3 className="h-5 w-5" />Performance Distribution</CardTitle>
							</CardHeader>
							<CardContent>
								<div className="space-y-4">
									<div className="flex justify-between items-center"><span className="text-sm text-muted-foreground no-select">Excellent (90-100%)</span><div className="flex items-center gap-2"><div className="w-24 bg-secondary rounded-full h-2"><div className="bg-green-500 h-2 rounded-full meter-fill" data-width="16.7%" style={{ width: 0 }}></div></div><span className="text-sm font-medium text-foreground no-select">25</span></div></div>
									<div className="flex justify-between items-center"><span className="text-sm text-muted-foreground no-select">Good (75-89%)</span><div className="flex items-center gap-2"><div className="w-24 bg-secondary rounded-full h-2"><div className="bg-blue-500 h-2 rounded-full meter-fill" data-width="40%" style={{ width: 0 }}></div></div><span className="text-sm font-medium text-foreground no-select">60</span></div></div>
									<div className="flex justify-between items-center"><span className="text-sm text-muted-foreground no-select">Average (60-74%)</span><div className="flex items-center gap-2"><div className="w-24 bg-secondary rounded-full h-2"><div className="bg-yellow-500 h-2 rounded-full meter-fill" data-width="30%" style={{ width: 0 }}></div></div><span className="text-sm font-medium text-foreground no-select">45</span></div></div>
									<div className="flex justify-between items-center"><span className="text-sm text-muted-foreground no-select">Needs Improvement (&lt;60%)</span><div className="flex items-center gap-2"><div className="w-24 bg-secondary rounded-full h-2"><div className="bg-red-500 h-2 rounded-full meter-fill" data-width="12%" style={{ width: 0 }}></div></div><span className="text-sm font-medium text-foreground no-select">18</span></div></div>
								</div>
							</CardContent>
						</Card>

						<Card className="no-select performance-card">
							<CardHeader>
								<CardTitle className="flex items-center gap-2 no-select"><PieChart className="h-5 w-5" />Evaluation Status</CardTitle>
							</CardHeader>
							<CardContent>
								<div className="space-y-4">
									<div className="text-center"><div className="text-3xl font-bold text-foreground no-select">{overviewStats.completedEvaluations}</div><div className="text-sm text-muted-foreground no-select">Total Evaluations</div></div>
									<div className="space-y-3">
										<div className="flex justify-between items-center"><div className="flex items-center gap-2"><div className="w-3 h-3 bg-green-500 rounded-full"></div><span className="text-sm text-muted-foreground no-select">Completed</span></div><span className="text-sm font-medium text-foreground no-select">{overviewStats.completedEvaluations}</span></div>
										<div className="flex justify-between items-center"><div className="flex items-center gap-2"><div className="w-3 h-3 bg-yellow-500 rounded-full"></div><span className="text-sm text-muted-foreground no-select">Pending</span></div><span className="text-sm font-medium text-foreground no-select">{overviewStats.pendingEvaluations}</span></div>
										<div className="flex justify-between items-center"><div className="flex items-center gap-2"><div className="w-3 h-3 bg-blue-500 rounded-full"></div><span className="text-sm text-muted-foreground no-select">In Progress</span></div><span className="text-sm font-medium text-foreground no-select">23</span></div>
									</div>
								</div>
							</CardContent>
						</Card>
					</div>

					<Card className="no-select">
						<CardHeader>
							<CardTitle className="no-select flex items-center gap-2"><Clock className="h-5 w-5 text-emerald-600" />Processing Progress Timeline</CardTitle>
						</CardHeader>
						<CardContent>
							<div ref={timelineRef} className="relative">
								<div className="absolute left-6 top-0 bottom-0 w-0.5 bg-emerald-500 timeline-line"></div>
								<div className="space-y-6 relative">
									<div className="flex items-center"><div className="w-3 h-3 bg-emerald-500 rounded-full z-10 relative timeline-point"></div><div className="ml-6 flex-1"><div className="flex justify-between items-center"><span className="text-sm font-medium text-foreground no-select">0 min</span><span className="text-lg font-bold text-emerald-600 no-select">0</span></div></div></div>
									<div className="flex items-center"><div className="w-3 h-3 bg-emerald-500 rounded-full z-10 relative timeline-point"></div><div className="ml-6 flex-1"><div className="flex justify-between items-center"><span className="text-sm font-medium text-foreground no-select">1 min</span><span className="text-lg font-bold text-emerald-600 no-select">850</span></div></div></div>
									<div className="flex items-center"><div className="w-3 h-3 bg-emerald-500 rounded-full z-10 relative timeline-point"></div><div className="ml-6 flex-1"><div className="flex justify-between items-center"><span className="text-sm font-medium text-foreground no-select">2 min</span><span className="text-lg font-bold text-emerald-600 no-select">1700</span></div></div></div>
									<div className="flex items-center"><div className="w-3 h-3 bg-emerald-500 rounded-full z-10 relative timeline-point"></div><div className="ml-6 flex-1"><div className="flex justify-between items-center"><span className="text-sm font-medium text-foreground no-select">3 min</span><span className="text-lg font-bold text-emerald-600 no-select">2550</span></div></div></div>
									<div className="flex items-center"><div className="w-3 h-3 bg-emerald-500 rounded-full z-10 relative timeline-point"></div><div className="ml-6 flex-1"><div className="flex justify-between items-center"><span className="text-sm font-medium text-foreground no-select">4 min</span><span className="text-lg font-bold text-emerald-600 no-select">3400</span></div></div></div>
								</div>
								<div className="mt-6 pt-6 border-t border-border">
									<div className="relative h-32 w-full">
										<svg className="w-full h-full" viewBox="0 0 400 120" preserveAspectRatio="none">
											<defs><pattern id="grid" width="80" height="30" patternUnits="userSpaceOnUse"><path d="M 80 0 L 0 0 0 30" fill="none" stroke="#e5e7eb" strokeWidth="0.5"/></pattern></defs>
											<rect width="100%" height="100%" fill="url(#grid)" />
											<polyline fill="none" stroke="#10b981" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" points="0,120 80,95 160,70 240,45 320,20 400,15" className="drop-shadow-sm progress-line" />
											<circle cx="0" cy="120" r="4" fill="#10b981" className="drop-shadow-sm"/>
											<circle cx="80" cy="95" r="4" fill="#10b981" className="drop-shadow-sm"/>
											<circle cx="160" cy="70" r="4" fill="#10b981" className="drop-shadow-sm"/>
											<circle cx="240" cy="45" r="4" fill="#10b981" className="drop-shadow-sm"/>
											<circle cx="320" cy="20" r="4" fill="#10b981" className="drop-shadow-sm"/>
											<circle cx="400" cy="15" r="4" fill="#10b981" className="drop-shadow-sm"/>
										</svg>
										<div className="absolute left-0 top-0 h-full flex flex-col justify-between text-xs text-muted-foreground py-1"><span className="no-select">3400</span><span className="no-select">2550</span><span className="no-select">1700</span><span className="no-select">850</span><span className="no-select">0</span></div>
										<div className="absolute bottom-0 left-0 w-full flex justify-between text-xs text-muted-foreground px-2"><span className="no-select">0 min</span><span className="no-select">1 min</span><span className="no-select">2 min</span><span className="no-select">3 min</span><span className="no-select">4 min</span></div>
									</div>
								</div>
								<div className="mt-6 pt-4 border-t border-border">
									<div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
										<div><div className="text-sm text-muted-foreground no-select">Total Processed</div><div className="text-xl font-bold text-emerald-600 no-select">3,400</div></div>
										<div><div className="text-sm text-muted-foreground no-select">Processing Rate</div><div className="text-xl font-bold text-emerald-600 no-select">850/min</div></div>
										<div><div className="text-sm text-muted-foreground no-select">Completion Time</div><div className="text-xl font-bold text-emerald-600 no-select">4 min</div></div>
									</div>
								</div>
								<div className="mt-6 pt-4 border-t border-border text-center"><p className="text-xs text-muted-foreground no-select">© 2024 Innomatics Research Labs. All rights reserved. • Automated OMR Evaluation System v2.0</p></div>
							</div>
						</CardContent>
					</Card>

					<Card className="no-select">
						<CardHeader>
							<CardTitle className="no-select">Key Insights & Recommendations</CardTitle>
						</CardHeader>
						<CardContent>
							<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
								<div className="p-4 bg-green-50 border border-green-200 rounded-lg"><div className="flex items-center gap-2 mb-2"><TrendingUp className="h-4 w-4 text-green-600" /><h4 className="font-medium text-green-800 no-select">High Performers</h4></div><p className="text-sm text-green-700 no-select">25 students are performing exceptionally well. Consider providing advanced material.</p></div>
								<div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg"><div className="flex items-center gap-2 mb-2"><Clock className="h-4 w-4 text-yellow-600" /><h4 className="font-medium text-yellow-800 no-select">Pending Evaluations</h4></div><p className="text-sm text-yellow-700 no-select">45 evaluations are still pending. Follow up with students for completion.</p></div>
								<div className="p-4 bg-red-50 border border-red-200 rounded-lg"><div className="flex items-center gap-2 mb-2"><AlertCircle className="h-4 w-4 text-red-600" /><h4 className="font-medium text-red-800 no-select">Needs Attention</h4></div><p className="text-sm text-red-700 no-select">18 students need additional support. Schedule remedial sessions.</p></div>
							</div>
						</CardContent>
					</Card>
				</TabsContent>
			</Tabs>
		</div>
	);
};

export default Overview;
