/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */
import { GoogleGenAI, Modality, Type } from '@google/genai';
import React, { useEffect, useState, useCallback, useRef, ChangeEvent } from 'react';
import ReactDOM from 'react-dom/client';
import JSZip from 'jszip';

const API_DELAY = 1000; // Reduced delay for a faster process
const LOCAL_STORAGE_KEY = 'photoNecromancerState';
const THEME_STORAGE_KEY = 'photoNecromancerTheme';
const TEST_MODE_STORAGE_KEY = 'photoNecromancerTestMode';


// REVERTED: The Genesis Prompt is now back to a more technical description of the app's function.
const GENESIS_PROMPT = `
Objective: Create a multi-step, AI-driven photo restoration application using the Gemini 2.5 flash-image model.

Core Process Flow:

1.  **Image Ingestion:** User uploads a source photograph. The application should display a preview of the uploaded image.

2.  **Initiate Processing:** Upon user confirmation, begin a two-stage deconstruction process. Provide real-time feedback and display generated assets as they become available.

3.  **Stage 1: Image Separation (2 API Calls):**
    *   **Subject Isolation:** Generate an image containing only the primary subjects (people, pets) on a pure white (#FFFFFF) background.
    *   **Background Inpainting:** Generate an image of the original background with the subjects removed and the resulting empty space filled generatively.

4.  **Stage 2: Asset Generation (11 API Calls):**
    *   **Character Sheet (1 Call):** From the subject isolation image, generate a photorealistic character turnaround sheet showing front, side, and back views for each subject, maintaining pose and expression.
    *   **Subject Maps (5 Calls):** From the subject isolation image, generate the following technical maps: Saliency, Normal, Depth, Edge, and Segmentation.
    *   **Background Maps (5 Calls):** From the background inpainting image, generate the same set of five technical maps.

5.  **Reconstruction Phase:** After all 13 deconstruction assets (2 separations + 1 sheet + 10 maps) are generated and displayed, present a new user control to begin the final restoration.

6.  **Final Restoration (10 API Calls):**
    *   Execute 10 separate, stateless API calls in sequence.
    *   For **each** call, provide the complete set of 13 generated deconstruction assets as context.
    *   Each call will use a unique, specific prompt from a predefined list to guide the AI in reconstructing the image with a different focus (e.g., color correction, scratch removal, upscaling).
    *   Introduce a delay between each of these 10 API calls to respect rate limits.

7.  **Results Display:**
    *   As each of the 10 restorations is completed, display it in a dedicated row.
    *   For each result, provide three comparison views: side-by-side, toggle, and an interactive slider.
    *   Include a dropdown in each result row that reveals the specific prompt and contextual assets used for that particular restoration.

8.  **UI/UX:**
    *   Implement a modern, clean interface with UI blur effects and a clear visual hierarchy.
    *   The user-facing text (buttons, titles, loading messages) should adopt a mature, humorous, and sarcastic tone. Instructional text (like this prompt and the "How It Works" guide) must remain clear and technical to avoid user confusion.
`;

const FACE_CLOSEUP_TITLE = "Identity-Preserving Close-up";

const MASTER_RESTORATION_PROMPTS = [
    { 
        title: "Comprehensive Restoration", 
        prompt: `Analyze the provided photograph and perform a comprehensive restoration. Your primary goal is to enhance the image to a professional quality while STRICTLY PRESERVING the original subject's identity, facial features, expression, and pose.
        Tasks to perform:
        1.  **Enhance Details:** Sharpen blurry areas, especially in faces, hair, and clothing textures, to bring out fine details without creating an artificial look. Synthesize plausible high-frequency detail where it's missing.
        2.  **Correct Colors:** Restore faded or shifted colors to a natural and vibrant state. Ensure skin tones are accurate and realistic.
        3.  **Repair Damage:** Seamlessly remove any scratches, dust, creases, or minor stains.
        4.  **Balance Lighting:** Adjust exposure, contrast, and shadows to create a balanced, well-lit image. Recover details from overexposed or underexposed areas.
        5.  **Reduce Noise:** Intelligently reduce digital noise and film grain while retaining essential image texture.
        The final output must be a clean, high-resolution, photorealistic image. DO NOT alter the composition or the subjects' core appearance.`
    },
    { 
        title: "Natural & Subtle Enhancement", 
        prompt: `Perform a subtle and natural restoration of this photograph. The key is to make it look like a well-preserved original, not an overly processed digital image. Adhere strictly to preserving the subject's identity, facial features, expression, and pose.
        Your focus should be on:
        1.  **Gentle Sharpening:** Apply a light touch to improve clarity without introducing harsh edges.
        2.  **Authentic Color:** Correct color casts and restore faded tones, aiming for a look that is true to the era of the photo.
        3.  **Minimal Repair:** Clean up distracting dust and minor scratches but leave the original film grain and texture intact.
        4.  **Soft Lighting:** Balance the light to feel natural and not overly dramatic.
        The result should be a tasteful, clean version of the original photo. DO NOT change the subjects or the overall mood.`
    },
    { 
        title: "Vibrant & Modern Remaster", 
        prompt: `Remaster this photograph with a vibrant, modern aesthetic. While you must strictly preserve the subject's identity, facial features, expression, and pose, the goal is to make the image pop with color and clarity.
        Execute the following:
        1.  **Crisp Details:** Maximize sharpness and texture detail, making every element clear and defined.
        2.  **Rich Colors:** Boost color saturation and vibrancy for a rich, contemporary feel, while keeping skin tones looking healthy and natural.
        3.  **Flawless Surface:** Remove all imperfections like scratches, noise, and grain for a perfectly clean look.
        4.  **Dynamic Contrast:** Enhance the contrast to create depth and a punchy, dynamic range between light and shadow.
        The output should be a high-impact, polished, and modern-looking photograph.`
    },
    { 
        title: "Cinematic & Artistic Grade", 
        prompt: `Apply a cinematic and artistic grade to this photograph, treating it like a still from a high-quality film. You must absolutely preserve the original subject's identity, facial features, expression, and pose.
        Your artistic direction is:
        1.  **Selective Focus:** Subtly enhance the sharpness of the subject to draw the viewer's eye, while keeping the background natural.
        2.  **Color Grading:** Apply a tasteful color grade to enhance the mood. You might shift the tones slightly towards warm or cool, or enhance a specific color palette within the image to create a more cohesive, artistic feel.
        3.  **Cinematic Lighting:** Adjust the lighting to add a touch of drama and depth, perhaps by deepening shadows or enhancing highlights, without losing detail.
        4.  **Perfect Blemishes:** Remove any distracting technical flaws (scratches, dust) while retaining the photo's inherent character.
        The final image should feel emotionally resonant and professionally graded, like a frame from a movie.`
    },
    {
        title: "Ultimate Quality Remaster",
        prompt: `Your task is to completely remaster the provided photograph, elevating it to the highest possible level of photorealism, as if it were captured with a top-of-the-line, professional-grade 2025 model Sony A7 series camera equipped with a G Master prime lens.

**Primary Objective:** Re-render the entire scene with breathtaking, hyper-realistic detail.

**Critical Constraint:** You MUST perfectly preserve the identity, likeness, and three-dimensional shape of the subjects' faces. Their facial structure, features, and expression are immutable. Do not alter their core appearance.

**Execution Details:**
1.  **Scene Reconstruction:** Recreate every object and background element from the original photo, but render them with ultra-high-definition textures, realistic materials, and perfect lighting.
2.  **Camera & Lens Simulation:** The final image must exhibit the characteristics of a high-end professional camera:
    *   **Clarity:** Edge-to-edge sharpness and micro-contrast. No digital noise or film grain.
    *   **Color:** Rich, vibrant, and accurate color science. Skin tones must be flawless and natural.
    *   **Lighting:** Natural and dynamic lighting with soft, detailed shadows and clean highlights.
    *   **Optics:** Simulate a shallow depth of field to create a beautiful separation between the subject and background, with creamy, aesthetically pleasing bokeh.
3.  **Overall Aesthetic:** The final output should be a pristine, jaw-droppingly detailed photograph that is indistinguishable from a modern, high-budget professional photoshoot.`
    }
];

const generateFaceTransplantPrompt = (faceDescriptions: string[]): string => {
    let prompt = `You are an expert digital artist performing a delicate face transplant. You have been given multiple images to work with:\n\n`;
    
    prompt += `- **The Base Image:** This is the main, high-quality photograph that the new faces should be integrated into. The faces in this image are the ones you will be replacing.\n\n`;
    
    prompt += `- **The Face Reference Images:** You have also been given ${faceDescriptions.length} separate close-up images. Each one is a highly detailed, accurate portrait of a specific person. The descriptions for these face references are:\n`;
    faceDescriptions.forEach((desc, index) => {
        prompt += `  - Face Reference ${index + 1}: A close-up of "${desc}".\n`;
    });
    prompt += `\n`;

    prompt += `Your task is to seamlessly replace EACH face in **The Base Image** with its corresponding, more accurate version from the **Face Reference Images**. For example, find the person in the base image who matches the description "${faceDescriptions[0]}" and replace their face with the corresponding reference image. Do this for all provided face references.\n\n`;
    
    prompt += `CRITICAL INSTRUCTIONS:\n`;
    prompt += `1.  The final output image MUST have the exact same dimensions, aspect ratio, and composition as **The Base Image**.\n`;
    prompt += `2.  Only modify the facial regions. Do not crop or alter any other part of the base image (background, clothing, etc.).\n`;
    prompt += `3.  Ensure the final image retains the overall high-fidelity lighting, color grading, and texture of **The Base Image** while perfectly integrating the identity and expression from each face reference. The transitions must be invisible.\n`;

    return prompt;
};

const FACE_COUNT_PROMPT = `Analyze the provided image and identify only the distinct human faces of the **primary, foreground subjects**. Ignore faces that are out of focus, in the distant background, or part of a crowd.

Return a JSON object with a single key "faces" which is an array of objects. For each identified primary face, provide:
1.  A brief, unique description that **MUST include their position in the frame** (e.g., "the woman on the far left").
2.  A "boundingBox" object with the pixel coordinates { "x": <left>, "y": <top>, "width": <width>, "height": <height> }.

Example response:
{
  "faces": [
    {
      "description": "the woman with blonde hair on the left",
      "boundingBox": { "x": 150, "y": 200, "width": 100, "height": 120 }
    }
  ]
}`;

const generateFaceCloseupPrompt = (description: string) => `From the original photograph provided, generate an extreme close-up, tightly cropped headshot of ONLY the person described as: "${description}".

CRITICAL INSTRUCTIONS:
1.  The subject's face MUST fill the entire frame, with nothing else but the same face, angled the same way, and white background showing in the places where there is no face. Do not show clothes, shoulders, arms, chest, or neck. The bottom of the same angled chin should be 5 pixels away from the bottom of the frame, and the top part of the original same angled hair should be 5 pixels from the top of the frame.
2.  The background MUST be pure white (#FFFFFF).
3.  You MUST perfectly preserve the subject's identity, facial structure, expression, and the angle of their head.
4.  **The output must be a hyper-detailed, photorealistic image, matching the artistic style of a modern, professional-grade photograph.** This is crucial for ensuring it can be seamlessly blended into a remastered image later.`;


const LOADING_MESSAGES = [
    "Giving the AI a Red Bull and a copy of your photo.", "Teaching the pixels to love again. It's not going well.", "This is taking a while because the AI is watching cat videos on your dime.", "Our servers are powered by potatoes and regret. Please hold.", "If this works, it's a miracle. If it doesn't, it was your photo's fault.", "Definitely not uploading your photo to a stock image site. Wink.", "The AI is currently arguing with itself about the color beige. This could take a minute.", "Don't tell anyone, but we're just running this through an Instagram filter.", "Enhancing... with sarcasm and a hint of judgment.", "Our highly-trained AI monkeys are flinging code at the screen. One of them is bound to stick.", "Rummaging through the digital bargain bin for some spare pixels...", "Did you try turning it off and on again? We are.", "I've seen things you people wouldn't believe... and this JPEG. This is worse.", "Recalibrating the WTF-o-meter.", "Please hold, we're Googling how to fix this.", "The AI is drunk with power. And possibly cheap vodka.", "Buffing out the sadness.", "Just a heads-up, the AI thinks your uncle looks like a potato.", "This is fine. Everything is fine.", "Honestly, we're as surprised as you are when this works.", "Spinning up the hamster wheel that powers this whole operation.", "The AI is asking for a raise. We told it to get back to work.", "Consulting the ancient texts of Stack Overflow.", "Charging the 'Make It Pretty' laser.", "Please don't refresh. The intern who wrote this code cries easily."
];


interface RestorationAsset { title: string; prompt: string; data: string; }
interface FaceCloseupAsset extends RestorationAsset {
    id: number;
    description: string;
    originalCroppedData: string;
}
interface ImagePart { data: string; mimeType: string; }
interface ApiRequestLog { id: number; type: string; timestamp: string; }
type Theme = 'light' | 'dark';
type AppState = 'idle' | 'resuming' | 'error';

// NEW: Color palette interfaces
interface HSLColor { h: number; s: number; l: number; }
interface ThemePalette {
    gradientTop: string;
    gradientBottom: string;
    wave1: string;
    wave2: string;
    wave3: string;
    panelTint: string;
    titleAccent: string;
    waveStroke: string;
    buttonBg: string;
    apiTrackerBg: string;
}
interface FullPalette {
    dark: ThemePalette;
    light: ThemePalette;
}

const base64ToBlob = (base64: string, mimeType: string): Blob => {
    const byteCharacters = atob(base64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], { type: mimeType });
};

const downloadImage = (base64Data: string, filename: string, mimeType = 'image/png') => {
    const blob = base64ToBlob(base64Data, mimeType);
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
};

const downloadImagesAsZip = async (images: { filename: string; base64: string }[], zipFilename: string) => {
    const zip = new JSZip();
    images.forEach(image => {
        zip.file(image.filename, image.base64, { base64: true });
    });
    const content = await zip.generateAsync({ type: 'blob' });
    const url = URL.createObjectURL(content);
    const a = document.createElement('a');
    a.href = url;
    a.download = zipFilename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
};

const fileToBase64 = (file: File): Promise<string> => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
        if (typeof reader.result === 'string') {
            resolve(reader.result.split(',')[1]);
        } else {
            reject(new Error('Failed to read file as a data URL.'));
        }
    };
    reader.onerror = error => reject(error);
});

const cropImage = (imageBase64: string, mimeType: string, boundingBox: { x: number; y: number; width: number; height: number; }): Promise<string> => {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            const canvas = document.createElement('canvas');
            const size = Math.max(boundingBox.width, boundingBox.height);
            canvas.width = size;
            canvas.height = size;
            const ctx = canvas.getContext('2d');
            if (!ctx) return reject(new Error('Could not get canvas context'));
            
            const sourceX = boundingBox.x + boundingBox.width / 2 - size / 2;
            const sourceY = boundingBox.y + boundingBox.height / 2 - size / 2;

            ctx.drawImage(img, sourceX, sourceY, size, size, 0, 0, size, size);
            resolve(canvas.toDataURL('image/jpeg').split(',')[1]);
        };
        img.onerror = reject;
        img.src = `data:${mimeType};base64,${imageBase64}`;
    });
};

const generateMockImage = (text: string): Promise<string> => {
    return new Promise((resolve) => {
        const canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 512;
        const ctx = canvas.getContext('2d');
        if (!ctx) return resolve('');

        ctx.fillStyle = '#333';
        ctx.fillRect(0, 0, 512, 512);

        ctx.fillStyle = '#fff';
        ctx.font = 'bold 24px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        
        const words = text.split(' ');
        let line = '';
        const yStart = 240;
        let lineCount = 0;
        const lineHeight = 30;
        ctx.fillText('[TEST MODE]', 256, 40);

        for(let n = 0; n < words.length; n++) {
          const testLine = line + words[n] + ' ';
          const metrics = ctx.measureText(testLine);
          const testWidth = metrics.width;
          if (testWidth > 480 && n > 0) {
            ctx.fillText(line, 256, yStart + (lineCount * lineHeight));
            line = words[n] + ' ';
            lineCount++;
          }
          else {
            line = testLine;
          }
        }
        ctx.fillText(line, 256, yStart + (lineCount * lineHeight));
        
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 4;
        ctx.strokeRect(0, 0, 512, 512);

        resolve(canvas.toDataURL('image/png').split(',')[1]);
    });
};

const generateDownloadFilename = (asset: RestorationAsset | FaceCloseupAsset): string => {
    let baseName = '';
    if ('description' in asset && asset.description) {
        // For face closeups, use the descriptive name
        baseName = `closeup_${asset.description}`;
    } else {
        // For other restorations, use the title
        baseName = asset.title;
    }
    // Sanitize the filename to be URL-friendly
    return `${baseName.replace(/[^a-z0-9_]/gi, '_').replace(/_{2,}/g, '_').toLowerCase()}.png`;
};

const PromptViewer = ({ prompt, title = "Show The Guts" }: { prompt: string; title?: string }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="prompt-viewer">
      <button onClick={() => setIsOpen(!isOpen)} className="prompt-toggle">
        {isOpen ? 'Hide The Guts' : title}
      </button>
      {isOpen && (
        <div className="prompt-content">
          {prompt}
        </div>
      )}
    </div>
  );
};

const ZoomableImage = ({ src, alt, onSingleTap }: { src: string; alt: string; onSingleTap?: () => void }) => {
    const [transform, setTransform] = useState({ scale: 1, x: 0, y: 0 });
    const isDragging = useRef(false);
    const hasDragged = useRef(false);
    const startPos = useRef({ x: 0, y: 0 });
    const containerRef = useRef<HTMLDivElement>(null);

    const handleWheel = (e: React.WheelEvent) => {
        e.preventDefault();
        const { deltaY } = e;
        const scaleAmount = -deltaY * 0.001;
        
        setTransform(prev => {
            const newScale = Math.max(1, Math.min(5, prev.scale + scaleAmount));
            if (newScale === 1) {
                return { scale: 1, x: 0, y: 0 };
            }
            return { ...prev, scale: newScale };
        });
    };

    const startDrag = (clientX: number, clientY: number) => {
        if (transform.scale <= 1) return;
        isDragging.current = true;
        hasDragged.current = false;
        startPos.current = { x: clientX - transform.x, y: clientY - transform.y };
    };

    const doDrag = (clientX: number, clientY: number) => {
        if (!isDragging.current || transform.scale <= 1) return;
        hasDragged.current = true;
        const newX = clientX - startPos.current.x;
        const newY = clientY - startPos.current.y;
        setTransform(prev => ({ ...prev, x: newX, y: newY }));
    };
    
    const endDrag = () => {
        // A short delay helps distinguish between a drag-end and a click.
        setTimeout(() => {
            isDragging.current = false;
        }, 50);
    };
    
    const handleMouseDown = (e: React.MouseEvent) => {
        e.preventDefault();
        startDrag(e.clientX, e.clientY);
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        if (!isDragging.current) return;
        e.preventDefault();
        doDrag(e.clientX, e.clientY);
    };

    const handleMouseUp = (e: React.MouseEvent) => {
        e.preventDefault();
        if (!hasDragged.current && onSingleTap) {
            onSingleTap();
        }
        endDrag();
    };

    const handleTouchStart = (e: React.TouchEvent) => {
        if (e.touches.length === 1) {
            startDrag(e.touches[0].clientX, e.touches[0].clientY);
        }
    };

    const handleTouchMove = (e: React.TouchEvent) => {
        if (e.touches.length === 1) {
            doDrag(e.touches[0].clientX, e.touches[0].clientY);
        }
    };

    const handleTouchEnd = (e: React.TouchEvent) => {
        if (!hasDragged.current && onSingleTap) {
            onSingleTap();
        }
        endDrag();
    };

    const handleDoubleClick = () => {
        setTransform({ scale: 1, x: 0, y: 0 });
    };

    return (
        <div 
            ref={containerRef}
            className="zoomable-container"
            onWheel={handleWheel}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={endDrag}
            onTouchStart={handleTouchStart}
            onTouchMove={handleTouchMove}
            onTouchEnd={handleTouchEnd}
            onDoubleClick={handleDoubleClick}
            title="Scroll to zoom, drag to pan, double-click to reset"
        >
            <img 
                src={src} 
                alt={alt} 
                style={{
                    transform: `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})`,
                    cursor: transform.scale > 1 ? (isDragging.current ? 'grabbing' : 'grab') : 'zoom-in',
                    transition: isDragging.current ? 'none' : 'transform 0.1s ease-out',
                }}
            />
        </div>
    );
};

const FaceComparisonSlider = ({ original, restored }: { original: string; restored: string; }) => {
    const [sliderPos, setSliderPos] = useState(50);
    const handleSliderMove = (e: React.MouseEvent<HTMLDivElement> | React.TouchEvent<HTMLDivElement>) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const x = 'touches' in e ? e.touches[0].clientX : e.clientX;
        const pos = ((x - rect.left) / rect.width) * 100;
        setSliderPos(Math.max(0, Math.min(100, pos)));
    };

    return (
        <div className="slider-wrapper" onMouseMove={handleSliderMove} onTouchMove={handleSliderMove}>
            <img className="slider-image-bottom" src={`data:image/jpeg;base64,${original}`} alt="Original Face" />
            <div className="slider-image-top" style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}>
                <img src={`data:image/png;base64,${restored}`} alt="Restored Face" />
            </div>
            <div className="slider-handle" style={{ left: `${sliderPos}%` }}></div>
        </div>
    );
};


const ComparisonViews = ({ original, restored, mimeType, downloadFilename }: { original: string; restored: string; mimeType: string, downloadFilename: string }) => {
    const [showOriginal, setShowOriginal] = useState(true);
    const [sliderPos, setSliderPos] = useState(50);

    const handleSliderMove = (e: React.MouseEvent<HTMLDivElement> | React.TouchEvent<HTMLDivElement>) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const x = 'touches' in e ? e.touches[0].clientX : e.clientX;
        const pos = ((x - rect.left) / rect.width) * 100;
        setSliderPos(Math.max(0, Math.min(100, pos)));
    };

    return (
        <div className="comparison-views-grid">
            <div className="comparison-view-item">
                <h4>Side-by-Side</h4>
                <div className="comparison-container">
                    <div className="image-wrapper">
                        <h5>The Original Sin</h5>
                        <ZoomableImage src={`data:${mimeType};base64,${original}`} alt="Original" />
                    </div>
                    <div className="image-wrapper">
                        <h5>The Glorious B*stard</h5>
                         <div className="downloadable-image" onClick={() => downloadImage(restored, downloadFilename)}>
                            <ZoomableImage src={`data:image/png;base64,${restored}`} alt="Restored" />
                            <div className="download-overlay">
                                <span>⬇️ Download</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div className="comparison-view-item">
                <h4>Toggle</h4>
                 <div className="toggle-wrapper">
                    <ZoomableImage 
                      src={showOriginal ? `data:${mimeType};base64,${original}` : `data:image/png;base64,${restored}`} 
                      alt="Toggled view"
                    />
                </div>
                <div className="toggle-buttons">
                    <button 
                        className={`toggle-button ${showOriginal ? 'active' : ''}`}
                        onClick={() => setShowOriginal(true)}
                        aria-pressed={showOriginal}
                    >
                        Before
                    </button>
                    <button 
                        className={`toggle-button ${!showOriginal ? 'active' : ''}`}
                        onClick={() => setShowOriginal(false)}
                        aria-pressed={!showOriginal}
                    >
                        After
                    </button>
                </div>
            </div>

            <div className="comparison-view-item">
                <h4>Slider</h4>
                 <div className="slider-wrapper" onMouseMove={handleSliderMove} onTouchMove={handleSliderMove}>
                    <img className="slider-image-bottom" src={`data:${mimeType};base64,${original}`} alt="Original" />
                    <div className="slider-image-top" style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}>
                         <img src={`data:image/png;base64,${restored}`} alt="Restored" />
                    </div>
                    <div className="slider-handle" style={{ left: `${sliderPos}%` }}></div>
                </div>
            </div>
        </div>
    );
};

const ExpandableSection = ({ title, children, defaultOpen = false, level = 0, actions }: { title: string, children?: React.ReactNode, defaultOpen?: boolean, level?: number, actions?: React.ReactNode }) => {
    const [isOpen, setIsOpen] = useState(defaultOpen);

    return (
        <div className={`expandable-section level-${level} ${isOpen ? 'open' : ''}`}>
            <div className="expandable-toggle-wrapper">
                <button onClick={() => setIsOpen(!isOpen)} className="expandable-toggle">
                    <span className="toggle-icon">►</span> {title}
                </button>
                {actions && <div className="expandable-actions">{actions}</div>}
            </div>
            <div className={`expandable-content ${isOpen ? 'open' : ''}`}>
                <div className="expandable-content-inner">
                    {children}
                </div>
            </div>
        </div>
    );
};


// REWRITTEN: The "How It Works" section now describes the new, efficient process.
const HowItWorks = () => {
    return (
        <div className="how-it-works-container">
            <ExpandableSection title="What The Hell Is This App?" defaultOpen={false}>
                <p><strong>This application performs a multi-stage, AI-driven photo restoration process.</strong></p>
                <p>It uses a powerful AI to generate multiple, unique restorations of a source image, each with a different artistic direction. This guide explains the new, more efficient workflow.</p>

                <ExpandableSection title="The New Restoration Process" level={1}>
                    <p>The following steps are executed to analyze and reconstruct the source image.</p>
                    
                    <ExpandableSection title="Step 1: Image Upload" level={2}>
                        <p>The process begins when a user uploads a source image. The system validates the file and prepares it for the AI.</p>
                    </ExpandableSection>
                    
                    <ExpandableSection title="Step 2: AI-Powered Remastering (6 API Calls)" level={2}>
                        <p>Instead of a complex 23-step deconstruction, the app now uses a direct and powerful "AI Art Director" approach.</p>
                        <ul>
                            <li><strong>Master Prompts:</strong> The system sends the original photo to the AI six separate times.</li>
                            <li><strong>Artistic Direction:</strong> Each of the six requests is paired with a unique and comprehensive "master prompt" that gives the AI a different artistic goal (e.g., "Comprehensive Restoration", "Vibrant & Modern Remaster").</li>
                            <li><strong>Identity Preservation:</strong> Every prompt strictly commands the AI to preserve the subject's identity, pose, and facial features, achieving the main goal of the old, complex pipeline with far greater efficiency.</li>
                            <li><strong>Direct Generation:</strong> The AI analyzes the original photo and generates six unique, high-quality remastered versions based on these instructions.</li>
                        </ul>
                    </ExpandableSection>
                    
                    <ExpandableSection title="Step 3: Results and Comparison" level={2}>
                        <p>The six generated restorations are displayed as they are completed.</p>
                        <ExpandableSection title="Interactive Comparison" level={3}>
                            <p>Each result is presented alongside the original image with three distinct comparison tools: a side-by-side view, a toggle view, and an interactive slider.</p>
                        </ExpandableSection>
                         <ExpandableSection title="Prompt Inspection" level={3}>
                             <p>Each result includes a dropdown viewer allowing the user to inspect the specific master prompt used to generate that image.</p>
                        </ExpandableSection>
                    </ExpandableSection>
                </ExpandableSection>
            </ExpandableSection>
        </div>
    );
};

// REWRITTEN: The ResultRow component is now collapsible for performance.
const ResultRow: React.FC<{
    result: RestorationAsset;
    originalImage: string;
    originalMimeType: string;
}> = ({ result, originalImage, originalMimeType }) => {
    
    const downloadFilename = generateDownloadFilename(result);

    return (
        <div className="result-row">
            <ExpandableSection 
                title={result.title} 
                defaultOpen={true}
                level={0}
                actions={
                    <button onClick={() => downloadImage(result.data, downloadFilename)} className="button-small">
                        Download Result
                    </button>
                }
            >
                <ComparisonViews 
                    original={originalImage} 
                    restored={result.data} 
                    mimeType={originalMimeType}
                    downloadFilename={downloadFilename}
                />

                <PromptViewer 
                  prompt={result.prompt} 
                  title={result.title === 'Unholy Face Transplant' ? 'Show The Transplant Prompt' : 'Show The Master Prompt'}
                />
            </ExpandableSection>
        </div>
    );
};



const ApiRequestTracker = ({ log }: { log: ApiRequestLog[] }) => {
    const [isOpen, setIsOpen] = useState(false);
    const totalRequests = log.length;
    const scrollRef = useRef<HTMLUListElement>(null);
    const [isGlowing, setIsGlowing] = useState(false);
    const prevTotalRequests = useRef(totalRequests);

    useEffect(() => {
        if (isOpen && scrollRef.current) {
            scrollRef.current.scrollTop = 0;
        }
    }, [log, isOpen]);

    useEffect(() => {
        if (totalRequests > prevTotalRequests.current) {
            setIsGlowing(true);
            const timer = setTimeout(() => setIsGlowing(false), 1000); // Glow animation duration
            return () => clearTimeout(timer);
        }
        prevTotalRequests.current = totalRequests;
    }, [totalRequests]);

    return (
        <div className={`api-tracker ${isOpen ? 'open' : ''}`}>
            <button
              onClick={() => setIsOpen(!isOpen)}
              className={`tracker-toggle ${isGlowing ? 'glowing' : ''}`}
              aria-label={`API Requests: ${totalRequests}`}>
              {totalRequests}
            </button>
            <div className="tracker-content">
                <h4>AI Shenanigans ({totalRequests})</h4>
                <ul ref={scrollRef}>
                    {log.slice().reverse().map(entry => (
                        <li key={entry.id}>
                            <span>{entry.id}. {entry.type}</span>
                            <small>{entry.timestamp}</small>
                        </li>
                    ))}
                </ul>
            </div>
        </div>
    );
};

interface TopBarProps {
    isProcessing: boolean;
    hasResults: boolean;
    hasSavedSession: boolean;
    onSave: () => void;
    onLoad: () => void;
    onDownloadAll: () => void;
    theme: Theme;
    onToggleTheme: () => void;
    isTestMode: boolean;
    onToggleTestMode: () => void;
}

const TopBar: React.FC<TopBarProps> = ({ isProcessing, hasResults, hasSavedSession, onSave, onLoad, onDownloadAll, theme, onToggleTheme, isTestMode, onToggleTestMode }) => {
    
    return (
        <div className="top-bar">
            <div className="top-bar-content">
                <div className="top-bar-title">Bad Photo Rehab</div>
                <div className="top-bar-actions">
                    <button 
                        className="button-small" 
                        onClick={onSave} 
                        disabled={isProcessing || !hasResults}
                        title={isProcessing ? 'Cannot save while processing' : !hasResults ? 'Nothing to save yet' : 'Save current session'}
                        aria-label="Save current session"
                    >
                        <span className="icon">💾</span>
                        <span className="text">Save My Trainwreck</span>
                    </button>
                    <button 
                        className="button-small" 
                        onClick={onLoad} 
                        disabled={!hasSavedSession || isProcessing}
                        title={!hasSavedSession ? 'No saved session found' : isProcessing ? 'Cannot load while processing' : 'Load saved session'}
                        aria-label="Load saved session"
                    >
                        <span className="icon">📂</span>
                        <span className="text">Load My Last Mistake</span>
                    </button>
                    <button 
                        className="button-small" 
                        onClick={onDownloadAll} 
                        disabled={!hasResults}
                        title={!hasResults ? 'Generate some images to download them' : 'Download all generated images as .zip'}
                        aria-label="Download all generated images as .zip"
                    >
                        <span className="icon">📦</span>
                        <span className="text">Grab All Crap (.zip)</span>
                    </button>
                    <button className={`test-mode-toggle ${isTestMode ? 'active' : ''}`} onClick={onToggleTestMode} title={`Test Mode is ${isTestMode ? 'ON' : 'OFF'}`}>
                        🧪
                    </button>
                    <button className="theme-toggle" onClick={onToggleTheme} title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}>
                        {theme === 'dark' ? '☀️' : '🌙'}
                    </button>
                </div>
            </div>
        </div>
    );
};

// NEW: Animated background component
const WavyBackground = ({ colorPalette }: { colorPalette: ThemePalette | null }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        let frameId: number;
        let time = 0;

        const resizeCanvas = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        };

        const draw = () => {
            if (!ctx || !colorPalette) {
                frameId = requestAnimationFrame(draw);
                return;
            };
            time += 0.002;
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const waves = [
                { yOffset: canvas.height * 0.15, amp: canvas.height * 0.1, freq: 0.02 / 1.5, speed: 1, offset: 0.5 },
                { yOffset: canvas.height * 0.4, amp: canvas.height * 0.15, freq: 0.015 / 1.5, speed: -1.2, offset: 1.5 },
                { yOffset: canvas.height * 0.65, amp: canvas.height * 0.1, freq: 0.01 / 1.5, speed: 1.5, offset: 3 },
                { yOffset: canvas.height * 0.9, amp: canvas.height * 0.12, freq: 0.008 / 1.5, speed: -1, offset: 4.5 }
            ];
            
            const strokeBaseColor = colorPalette.waveStroke;

            waves.forEach((wave, index) => {
                const wavePath = new Path2D();
                let minY = canvas.height;
                let maxY = 0;

                for (let x = 0; x <= canvas.width; x++) {
                    const y = wave.yOffset + wave.amp * Math.sin(x * wave.freq + time * wave.speed + wave.offset);
                    if (x === 0) wavePath.moveTo(x, y);
                    else wavePath.lineTo(x, y);
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
                
                const isDownward = index % 2 === 0;

                const fillPath = new Path2D(wavePath);
                if (isDownward) {
                    fillPath.lineTo(canvas.width, canvas.height);
                    fillPath.lineTo(0, canvas.height);
                } else { // Is upward
                    fillPath.lineTo(canvas.width, 0);
                    fillPath.lineTo(0, 0);
                }
                fillPath.closePath();
                
                const gradientHeight = wave.amp * 2;
                const gradient = isDownward
                    ? ctx.createLinearGradient(0, minY - wave.amp * 0.1, 0, minY + gradientHeight)
                    : ctx.createLinearGradient(0, maxY + wave.amp * 0.1, 0, maxY - gradientHeight);

                const strokeFillStart = strokeBaseColor.replace(/hsla?\(([^,]+),([^,]+),([^,]+),[^)]+\)/, `hsla($1,$2,$3,0.45)`);
                const strokeFillEnd = strokeBaseColor.replace(/hsla?\(([^,]+),([^,]+),([^,]+),[^)]+\)/, `hsla($1,$2,$3,0)`);
                
                gradient.addColorStop(0, strokeFillStart);
                gradient.addColorStop(1, strokeFillEnd);
                
                ctx.fillStyle = gradient;
                ctx.fill(fillPath);
                
                // PERFORMANCE OPTIMIZATION: Removed costly per-frame canvas filter.
                // The main canvas element has a CSS blur which is much more efficient.
                ctx.strokeStyle = colorPalette.waveStroke;
                ctx.lineWidth = 2;
                ctx.stroke(wavePath);
            });
            
            frameId = requestAnimationFrame(draw);
        };

        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);
        draw();

        return () => {
            cancelAnimationFrame(frameId);
            window.removeEventListener('resize', resizeCanvas);
        };
    }, [colorPalette]);

    return <canvas id="background-canvas" ref={canvasRef}></canvas>;
};


function App() {
  const [appState, setAppState] = useState<AppState>('idle');
  const [isDisclaimerVisible, setIsDisclaimerVisible] = useState(true);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [originalMimeType, setOriginalMimeType] = useState('');
  const [restorationResults, setRestorationResults] = useState<RestorationAsset[]>([]);
  const [faceCloseups, setFaceCloseups] = useState<FaceCloseupAsset[]>([]);
  const [processingFaceIds, setProcessingFaceIds] = useState<number[]>([]);
  const [isTransplanting, setIsTransplanting] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [loadingMessage, setLoadingMessage] = useState(LOADING_MESSAGES[0]);
  const [apiRequestLog, setApiRequestLog] = useState<ApiRequestLog[]>([]);
  const [hasSavedSession, setHasSavedSession] = useState(false);
  const [processingTitles, setProcessingTitles] = useState<string[]>([]);
  
  const [theme, setTheme] = useState<Theme>(() => {
    const savedTheme = localStorage.getItem(THEME_STORAGE_KEY);
    if (savedTheme === 'light' || savedTheme === 'dark') return savedTheme;
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) return 'light';
    return 'dark';
  });
  const [isTestMode, setIsTestMode] = useState<boolean>(() => {
    return localStorage.getItem(TEST_MODE_STORAGE_KEY) === 'true';
  });
  const [colorPalette, setColorPalette] = useState<FullPalette | null>(null);
  
  const loadingIntervalRef = useRef<number | null>(null);
  const resultsEndRef = useRef<HTMLDivElement>(null);

  const toggleTheme = useCallback(() => {
    setTheme(prevTheme => prevTheme === 'dark' ? 'light' : 'dark');
  }, []);

  const toggleTestMode = useCallback(() => {
    setIsTestMode(prev => {
        const newState = !prev;
        localStorage.setItem(TEST_MODE_STORAGE_KEY, String(newState));
        alert(`Test Mode is now ${newState ? 'ON' : 'OFF'}. API calls will be ${newState ? 'simulated' : 'real'}.`);
        return newState;
    });
  }, []);

  useEffect(() => {
    document.documentElement.className = `theme-${theme}`;
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);
  
    const extractColorsFromImage = useCallback(async (img: HTMLImageElement): Promise<FullPalette> => {
        return new Promise<FullPalette>((resolve) => {
            const defaultDark: ThemePalette = { gradientTop: '#2a2b36', gradientBottom: '#1a1b26', wave1: '#7aa2f7', wave2: '#bb9af7', wave3: '#f7768e', panelTint: 'rgba(40, 42, 54, 0.4)', titleAccent: '#bb9af7', waveStroke: 'rgba(255, 255, 255, 0.7)', buttonBg: '#7aa2f7', apiTrackerBg: '#f7768e' };
            const defaultLight: ThemePalette = { gradientTop: '#e0e4f0', gradientBottom: '#c0c5d4', wave1: '#5a7dbd', wave2: '#8f72c3', wave3: '#d35c71', panelTint: 'rgba(255, 255, 255, 0.4)', titleAccent: '#5a7dbd', waveStroke: 'rgba(90, 125, 189, 0.9)', buttonBg: '#5a7dbd', apiTrackerBg: '#d35c71' };
            const defaultPalette: FullPalette = { dark: defaultDark, light: defaultLight };

            const canvas = document.createElement('canvas');
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            const ctx = canvas.getContext('2d', { willReadFrequently: true });
            if (!ctx) return resolve(defaultPalette);
            
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

            const colorCounts: { [key: string]: number } = {};
            const samplePoints = 15; // Sample a 15x15 grid
            for (let x = 0; x < samplePoints; x++) {
                for (let y = 0; y < samplePoints; y++) {
                    const px = Math.floor(canvas.width * (x + 0.5) / samplePoints);
                    const py = Math.floor(canvas.height * (y + 0.5) / samplePoints);
                    const [r, g, b] = ctx.getImageData(px, py, 1, 1).data;
                    
                    const max = Math.max(r, g, b), min = Math.min(r, g, b);
                    if (max - min < 25) continue;
                    if ((r + g + b) < 40 || (r + g + b) > 720) continue;

                    const r_q = Math.round(r / 32) * 32;
                    const g_q = Math.round(g / 32) * 32;
                    const b_q = Math.round(b / 32) * 32;
                    const key = `${r_q},${g_q},${b_q}`;
                    colorCounts[key] = (colorCounts[key] || 0) + 1;
                }
            }

            const sortedColors = Object.entries(colorCounts).sort(([,a],[,b]) => b - a);
            
            const rgbToHsl = (r: number, g: number, b: number): HSLColor => {
                r /= 255; g /= 255; b /= 255;
                const max = Math.max(r, g, b), min = Math.min(r, g, b);
                let h=0, s=0;
                const l = (max + min) / 2;
                if (max !== min) {
                    const d = max - min;
                    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
                    if (max === r) h = (g - b) / d + (g < b ? 6 : 0);
                    else if (max === g) h = (b - r) / d + 2;
                    else h = (r - g) / d + 4;
                    h /= 6;
                }
                return { h: Math.round(h * 360), s: Math.round(s * 100), l: Math.round(l * 100) };
            };
            
            const topRgbStrings = sortedColors.slice(0, 4).map(entry => entry[0]);
            
            let topHues = topRgbStrings.map(rgbStr => {
                const [r,g,b] = rgbStr.split(',').map(Number);
                return rgbToHsl(r,g,b);
            });
            
            while (topHues.length < 4) {
                if (topHues.length === 0) return resolve(defaultPalette);
                const lastColor = topHues[topHues.length - 1];
                topHues.push({
                    h: (lastColor.h + 70 + Math.random() * 20) % 360,
                    s: Math.max(45, Math.min(85, lastColor.s + (Math.random() - 0.5) * 15)),
                    l: Math.max(45, Math.min(65, lastColor.l + (Math.random() - 0.5) * 15)),
                });
            }
            
            const toHslString = (c: HSLColor) => `hsl(${c.h}, ${c.s}%, ${c.l}%)`;
            const toHslaString = (c: HSLColor, a: number) => `hsla(${c.h}, ${c.s}%, ${c.l}%, ${a})`;
            const mainColor = topHues[0];
            const accent1 = topHues[1];
            const accent2 = topHues[2];
            const accent3 = topHues[3];

            const darkPalette: ThemePalette = {
                gradientBottom: toHslString({ h: mainColor.h, s: mainColor.s, l: 10 }),
                gradientTop: toHslString({ h: mainColor.h, s: mainColor.s, l: 20 }),
                wave1: toHslString({ h: mainColor.h, s: Math.min(100, mainColor.s + 10), l: 65 }),
                wave2: toHslString({ h: accent1.h, s: Math.min(100, accent1.s + 10), l: 70 }),
                wave3: toHslString({ h: accent2.h, s: Math.min(100, accent2.s + 10), l: 75 }),
                panelTint: `hsla(${accent3.h}, ${Math.round(accent3.s / 1.8)}%, ${Math.round(accent3.l / 1.2)}%, 0.25)`,
                titleAccent: toHslString({ h: accent1.h, s: Math.min(100, accent1.s + 30), l: 80 }),
                waveStroke: 'hsla(0, 0%, 100%, 0.7)',
                buttonBg: toHslString({ h: mainColor.h, s: Math.round(mainColor.s * 0.7), l: 45 }),
                apiTrackerBg: toHslString({ h: accent2.h, s: accent2.s, l: 55 }),
            };
            
            const lightPalette: ThemePalette = {
                gradientTop: toHslString({ h: mainColor.h, s: Math.min(95, mainColor.s + 20), l: 94 }),
                gradientBottom: toHslString({ h: mainColor.h, s: Math.min(95, mainColor.s + 20), l: 84 }),
                wave1: toHslString({ h: mainColor.h, s: Math.min(100, mainColor.s + 10), l: 65 }),
                wave2: toHslString({ h: accent1.h, s: Math.min(100, accent1.s + 10), l: 60 }),
                wave3: toHslString({ h: accent2.h, s: Math.min(100, accent2.s + 10), l: 70 }),
                panelTint: `hsla(${accent3.h}, ${Math.round(accent3.s / 1.5)}%, ${Math.round(accent3.l / 2)}%, 0.2)`,
                titleAccent: toHslString({ h: accent1.h, s: Math.min(100, accent1.s + 25), l: 35 }),
                waveStroke: toHslaString({ h: mainColor.h, s: Math.min(100, mainColor.s + 10), l: 40 }, 0.9),
                buttonBg: toHslString({ h: mainColor.h, s: Math.round(mainColor.s * 0.7), l: 50 }),
                apiTrackerBg: toHslString({ h: accent2.h, s: accent2.s, l: 55 }),
            };
            
            resolve({ dark: darkPalette, light: lightPalette });
        });
    }, []);

  useEffect(() => {
    if (colorPalette) {
        const currentThemePalette = colorPalette[theme];
        document.body.style.backgroundImage = `linear-gradient(to bottom, ${currentThemePalette.gradientTop}, ${currentThemePalette.gradientBottom})`;
        document.documentElement.style.setProperty('--accent-color', currentThemePalette.titleAccent);
        document.documentElement.style.setProperty('--button-bg-color', currentThemePalette.buttonBg);
        document.documentElement.style.setProperty('--api-tracker-bg-color', currentThemePalette.apiTrackerBg);
        
        const titleAccent = currentThemePalette.titleAccent;
        const hslValues = titleAccent.match(/(\d+)/g);
        if (hslValues && hslValues.length >= 3) {
            const [h, s, l] = hslValues.map(Number);
            const darkenedL = Math.max(0, l * 0.8); // Darken by 20%
            const topBarTintColor = `hsla(${h}, ${s}%, ${darkenedL}%, 0.15)`; // 15% strength
            document.documentElement.style.setProperty('--top-bar-tint', topBarTintColor);
        } else {
            document.documentElement.style.removeProperty('--top-bar-tint');
        }

    } else {
        const defaultBg = theme === 'dark'
            ? 'linear-gradient(to bottom, #2a2b36, #1a1b26)'
            : 'linear-gradient(to bottom, #e0e4f0, #c0c5d4)';
        document.body.style.backgroundImage = defaultBg;
        document.documentElement.style.removeProperty('--accent-color');
        document.documentElement.style.removeProperty('--button-bg-color');
        document.documentElement.style.removeProperty('--api-tracker-bg-color');
        document.documentElement.style.removeProperty('--top-bar-tint');
    }
  }, [colorPalette, theme]);

  useEffect(() => {
      const savedStateJSON = localStorage.getItem(LOCAL_STORAGE_KEY);
      if (savedStateJSON) {
          try {
              const savedState = JSON.parse(savedStateJSON);
              if (savedState.originalImage) {
                  setHasSavedSession(true);
              }
          } catch (e) {
              console.error("Failed to parse saved state, clearing it.", e);
              localStorage.removeItem(LOCAL_STORAGE_KEY);
          }
      }
  }, []);

  const handleSave = useCallback(() => {
    const stateToSave = {
        originalImage, originalMimeType, 
        restorationResults, faceCloseups,
        apiRequestLog, colorPalette, errorMessage,
    };
    if (originalImage) {
        localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(stateToSave));
        setHasSavedSession(true);
        alert('Session saved! Your glorious mess is safe.');
    }
  }, [
      originalImage, originalMimeType, 
      restorationResults, faceCloseups,
      apiRequestLog, colorPalette, errorMessage
  ]);

  const logApiRequest = useCallback((type: string) => {
    setApiRequestLog(prevLog => [
        ...prevLog,
        {
            id: prevLog.length + 1,
            type: isTestMode ? `[TEST] ${type}` : type,
            timestamp: new Date().toLocaleTimeString()
        }
    ]);
  }, [isTestMode]);

  useEffect(() => {
    const isProcessing = processingTitles.length > 0 || isTransplanting || processingFaceIds.length > 0;
    if (isProcessing) {
        loadingIntervalRef.current = window.setInterval(() => {
            setLoadingMessage(LOADING_MESSAGES[Math.floor(Math.random() * LOADING_MESSAGES.length)]);
        }, 3000);
    } else {
        if (loadingIntervalRef.current) clearInterval(loadingIntervalRef.current);
    }
    return () => {
        if (loadingIntervalRef.current) clearInterval(loadingIntervalRef.current);
    };
  }, [processingTitles, isTransplanting, processingFaceIds]);
  
    const handleReset = useCallback(() => {
        setAppState('idle');
        setIsDisclaimerVisible(false);
        setSelectedFile(null);
        setOriginalImage(null);
        setOriginalMimeType('');
        setRestorationResults([]);
        setFaceCloseups([]);
        setProcessingFaceIds([]);
        setIsTransplanting(false);
        setErrorMessage('');
        setColorPalette(null);
        setApiRequestLog([]);
        setProcessingTitles([]);
    }, []);

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      try {
        handleReset();
        setSelectedFile(file);
        setOriginalMimeType(file.type);
        const base64Data = await fileToBase64(file);
        setOriginalImage(base64Data);
        setErrorMessage(''); 
      } catch (error) {
        setErrorMessage('Failed to read the file. Maybe it\'s cursed? Or just... bad.');
        setAppState('error');
      }
    }
  };

  const generateImage = useCallback(async (prompt: string, image: ImagePart) => {
    if (isTestMode) {
        await new Promise(resolve => setTimeout(resolve, 750));
        const title = prompt.substring(0, 50).replace(/"/g, '') + '...';
        return generateMockImage(`Mock Image for: "${title}"`);
    }
    console.log(`Sending prompt: "${prompt.substring(0, 30)}..."`);
    const ai = new GoogleGenAI({apiKey: process.env.API_KEY});
    const parts = [
      { inlineData: { data: image.data, mimeType: image.mimeType } },
      { text: prompt },
    ];

    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash-image',
      contents: { parts },
      config: { responseModalities: [Modality.IMAGE] },
    });
    
    const responseParts = response.candidates?.[0]?.content?.parts;
    if (Array.isArray(responseParts)) {
        for (const part of responseParts) {
            if (part.inlineData?.data) {
                return part.inlineData.data;
            }
        }
    }
    throw new Error("The AI refused to create an image. It has standards, apparently.");
  }, [isTestMode]);

  const generateJson = useCallback(async (prompt: string, image: ImagePart) => {
    if (isTestMode) {
        await new Promise(resolve => setTimeout(resolve, 500));
        return {
            "faces": [
                {
                    "description": "mock person on the left",
                    "boundingBox": { "x": 120, "y": 150, "width": 200, "height": 240 }
                }
            ]
        };
    }
    console.log(`Sending JSON prompt: "${prompt.substring(0, 30)}..."`);
    const ai = new GoogleGenAI({apiKey: process.env.API_KEY});
    const parts = [
        { inlineData: { data: image.data, mimeType: image.mimeType } },
        { text: prompt },
    ];
    
    const responseSchema = {
        type: Type.OBJECT,
        properties: {
            faces: {
                type: Type.ARRAY,
                items: {
                    type: Type.OBJECT,
                    properties: {
                        description: { type: Type.STRING },
                        boundingBox: {
                            type: Type.OBJECT,
                            properties: {
                                x: { type: Type.INTEGER },
                                y: { type: Type.INTEGER },
                                width: { type: Type.INTEGER },
                                height: { type: Type.INTEGER },
                            },
                            required: ['x', 'y', 'width', 'height'],
                        }
                    },
                    required: ['description', 'boundingBox'],
                },
            },
        },
        required: ['faces'],
    };

    const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: { parts },
        config: { 
            responseMimeType: "application/json",
            responseSchema,
        },
    });

    try {
        const jsonText = response.text.trim();
        return JSON.parse(jsonText);
    } catch (e) {
        console.error("Failed to parse JSON response:", response.text);
        throw new Error("The AI returned malformed data when trying to count faces.");
    }
  }, [isTestMode]);
  
  const generateCompositeImage = useCallback(async (prompt: string, images: ImagePart[]) => {
      if (isTestMode) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        const title = prompt.substring(0, 50).replace(/"/g, '') + '...';
        return generateMockImage(`Mock Composite for: "${title}"`);
      }
      console.log(`Sending composite prompt: "${prompt.substring(0, 30)}..."`);
      const ai = new GoogleGenAI({apiKey: process.env.API_KEY});

      const imageParts = images.map(image => ({
          inlineData: { data: image.data, mimeType: image.mimeType }
      }));
      const textPart = { text: prompt };
      const parts = [...imageParts, textPart];

      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash-image',
        contents: { parts },
        config: { responseModalities: [Modality.IMAGE] },
      });
      
      const responseParts = response.candidates?.[0]?.content?.parts;
      if (Array.isArray(responseParts)) {
          for (const part of responseParts) {
              if (part.inlineData?.data) {
                  return part.inlineData.data;
              }
          }
      }
      throw new Error("The AI refused to create the composite image. It's probably an artist, you know?");
  }, [isTestMode]);

  const handleError = useCallback((err: any, title?: string) => {
    console.error(`Error during [${title || 'general'}] process:`, err);
    const errorMessage = err.message || "";
    let specificMessage = `The AI had a meltdown during the '${title}' generation. It mumbled something about "${errorMessage}" and started crying. Maybe try again?`;

    if (errorMessage.includes('429') || errorMessage.includes('Resource Exhausted')) {
        specificMessage = "Whoa there, cowboy! You're hitting the API harder than a piñata at a birthday party. The AI needs a smoke break. Give it a rest and try again tomorrow. Your daily limit is up, you magnificent bastard.";
    }
    setErrorMessage(specificMessage);
  }, []);

  const handleSingleRestoration = useCallback(async (promptItem: {title: string, prompt: string}) => {
    if (!originalImage || !originalMimeType) return;

    setProcessingTitles(prev => [...prev, promptItem.title]);
    setErrorMessage(''); // Clear previous errors on new attempt

    try {
        await new Promise(resolve => setTimeout(resolve, API_DELAY));
        const imagePart = { data: originalImage, mimeType: originalMimeType };
        const resultBase64 = await generateImage(promptItem.prompt, imagePart);
        logApiRequest(promptItem.title);
        setRestorationResults(prev => [...prev, { ...promptItem, data: resultBase64 }]);
    } catch (err: any) {
        handleError(err, promptItem.title);
    } finally {
        setProcessingTitles(prev => prev.filter(t => t !== promptItem.title));
    }
  }, [originalImage, originalMimeType, generateImage, logApiRequest, handleError]);

  const handleIterateRestoration = useCallback(async (title: string) => {
    const currentResult = restorationResults.find(r => r.title === title);
    if (!currentResult) {
        setErrorMessage("Couldn't find the result to iterate on. Something is very wrong.");
        return;
    }

    setProcessingTitles(prev => [...prev, title]);
    setErrorMessage('');

    try {
        await new Promise(resolve => setTimeout(resolve, API_DELAY));
        // The new input is the previous result, which is a PNG
        const imagePart = { data: currentResult.data, mimeType: 'image/png' };
        const resultBase64 = await generateImage(currentResult.prompt, imagePart);
        logApiRequest(`Iterate: ${title}`);
        setRestorationResults(prev => prev.map(r => 
            r.title === title ? { ...r, data: resultBase64 } : r
        ));
    } catch (err: any) {
        handleError(err, `Iterate: ${title}`);
    } finally {
        setProcessingTitles(prev => prev.filter(t => t !== title));
    }
  }, [restorationResults, generateImage, logApiRequest, handleError]);

  const handleGenerateAll = useCallback(async () => {
    const existingTitles = new Set(restorationResults.map(r => r.title));
    const titlesToProcess = MASTER_RESTORATION_PROMPTS.filter(p => !existingTitles.has(p.title));
    
    for (const promptItem of titlesToProcess) {
        await handleSingleRestoration(promptItem);
    }
  }, [restorationResults, handleSingleRestoration]);
  
    const handleGenerateFaceCloseups = useCallback(async () => {
        if (!originalImage || !originalMimeType) return;
        setErrorMessage('');
        setFaceCloseups([]); // Clear previous faces
        setProcessingTitles(prev => [...prev, FACE_CLOSEUP_TITLE]);

        try {
            const imagePart = { data: originalImage, mimeType: originalMimeType };
            logApiRequest("Detect Faces");
            const faceData = await generateJson(FACE_COUNT_PROMPT, imagePart);
            const faces: { description: string; boundingBox: { x: number, y: number, width: number, height: number } }[] = faceData.faces || [];
            
            if (faces.length === 0) {
                setErrorMessage("The AI couldn't find any faces to work with. Maybe it's shy?");
                setProcessingTitles(prev => prev.filter(t => t !== FACE_CLOSEUP_TITLE));
                return;
            }

            const facePromises = faces.map(async (face, index) => {
                const faceId = index + 1;
                setProcessingFaceIds(prev => [...prev, faceId]);
                try {
                    const originalCroppedData = await cropImage(originalImage, originalMimeType, face.boundingBox);
                    const prompt = generateFaceCloseupPrompt(face.description);
                    await new Promise(resolve => setTimeout(resolve, API_DELAY));
                    logApiRequest(`Face ${faceId} Close-up`);
                    const resultBase64 = await generateImage(prompt, imagePart);
                    const newFace: FaceCloseupAsset = {
                        id: faceId,
                        title: `Face ${faceId}: ${face.description}`,
                        prompt,
                        description: face.description,
                        data: resultBase64,
                        originalCroppedData,
                    };
                    setFaceCloseups(prev => [...prev, newFace].sort((a, b) => a.id - b.id));
                } catch (err: any) {
                    handleError(err, `Face ${faceId} Close-up`);
                } finally {
                    setProcessingFaceIds(prev => prev.filter(id => id !== faceId));
                }
            });

            await Promise.all(facePromises);

        } catch (err: any) {
            handleError(err, "Detect Faces");
        } finally {
            setProcessingTitles(prev => prev.filter(t => t !== FACE_CLOSEUP_TITLE));
        }
    }, [originalImage, originalMimeType, generateJson, generateImage, logApiRequest, handleError]);

    const handleRegenerateFace = useCallback(async (faceToRegen: FaceCloseupAsset) => {
        if (!originalImage || !originalMimeType) return;
        setErrorMessage('');
        setProcessingFaceIds(prev => [...prev, faceToRegen.id]);

        try {
            const imagePart = { data: originalImage, mimeType: originalMimeType };
            await new Promise(resolve => setTimeout(resolve, API_DELAY));
            logApiRequest(`Regen Face ${faceToRegen.id}`);
            const resultBase64 = await generateImage(faceToRegen.prompt, imagePart);
            
            setFaceCloseups(prev => prev.map(face => 
                face.id === faceToRegen.id ? { ...face, data: resultBase64 } : face
            ));

        } catch (err: any) {
            handleError(err, `Regen Face ${faceToRegen.id}`);
        } finally {
            setProcessingFaceIds(prev => prev.filter(id => id !== faceToRegen.id));
        }
    }, [originalImage, originalMimeType, generateImage, logApiRequest, handleError]);

    const handleFaceTransplant = useCallback(async (isIteration = false) => {
        const baseImageTitle = isIteration ? "Unholy Face Transplant" : "Ultimate Quality Remaster";
        const baseImage = restorationResults.find(r => r.title === baseImageTitle);

        if (!baseImage || faceCloseups.length === 0) {
            setErrorMessage(`Can't perform transplant. You need the '${baseImageTitle}' result and at least one face close-up.`);
            return;
        }
        
        setIsTransplanting(true);
        setErrorMessage('');
        try {
            const logType = isIteration ? "Iterate on Transplant" : "Unholy Face Transplant";
            logApiRequest(logType);
            
            const faceDescriptions = faceCloseups.map(face => face.description);
            const dynamicPrompt = generateFaceTransplantPrompt(faceDescriptions);

            const faceImageParts = faceCloseups.map(face => ({
                data: face.data,
                mimeType: 'image/png'
            }));

            const resultBase64 = await generateCompositeImage(
                dynamicPrompt,
                [
                    { data: baseImage.data, mimeType: 'image/png' },
                    ...faceImageParts
                ]
            );
            
            const newResult = {
                title: "Unholy Face Transplant",
                prompt: dynamicPrompt,
                data: resultBase64
            };

            if (isIteration) {
                setRestorationResults(prev => prev.map(r => r.title === "Unholy Face Transplant" ? newResult : r));
            } else {
                setRestorationResults(prev => [...prev, newResult]);
            }
        } catch (err: any) {
            handleError(err, "Unholy Face Transplant");
        } finally {
            setIsTransplanting(false);
        }
    }, [restorationResults, faceCloseups, generateCompositeImage, logApiRequest, handleError]);

  useEffect(() => {
    if (appState !== 'resuming') return;
    
    const resumeProcess = async () => {
        const savedStateJSON = localStorage.getItem(LOCAL_STORAGE_KEY);
        if (!savedStateJSON) {
            handleReset();
            return;
        }

        try {
            const savedState = JSON.parse(savedStateJSON);
            setOriginalImage(savedState.originalImage || null);
            setOriginalMimeType(savedState.originalMimeType || '');
            setColorPalette(savedState.colorPalette || null);
            setRestorationResults(savedState.restorationResults || []);
            setFaceCloseups(savedState.faceCloseups || []);
            setApiRequestLog(savedState.apiRequestLog || []);
            setErrorMessage(savedState.errorMessage || '');
            setIsDisclaimerVisible(false);
            setAppState('idle'); // Resuming is done, go back to idle state
        } catch (e) {
            console.error("Failed to resume state, starting fresh.", e);
            handleReset();
        }
    };
    
    const timer = setTimeout(resumeProcess, 100);
    return () => clearTimeout(timer);
  }, [appState, handleReset]);

  useEffect(() => {
      if (restorationResults.length > 0) {
          resultsEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
      }
  }, [restorationResults.length]);

  const handleLoad = () => {
      setAppState('resuming');
  };
  
  const handleDownloadAll = () => {
      const images: { filename: string; base64: string }[] = [];
      
      if (originalImage) {
          images.push({ filename: 'original_image.png', base64: originalImage });
      }

      restorationResults.forEach((result) => {
          const filename = `restored/${generateDownloadFilename(result)}`;
          images.push({ filename, base64: result.data });
      });

      faceCloseups.forEach((face) => {
          const filename = `closeups/${generateDownloadFilename(face)}`;
          images.push({ filename, base64: face.data });
      });

      downloadImagesAsZip(images, 'bad_photo_rehab_gallery.zip');
  };

  const panelStyle = colorPalette ? { backgroundColor: colorPalette[theme].panelTint } : {};
  const titleStyle = colorPalette ? { color: colorPalette[theme].titleAccent } : {};

  const renderUploader = () => (
    <div className="app-container" style={panelStyle}>
      <h1 style={titleStyle}>Bad Photo Rehab</h1>
      {isDisclaimerVisible && (
        <div className="disclaimer-box" onClick={() => setIsDisclaimerVisible(false)}>
          <p><strong>Warning, You Absolute Maniac!</strong></p>
          <p>This glorious piece of tech now makes about <strong>6 calls to the AI</strong> for every single photo. This is way more efficient, but the free API still has a daily limit (~1000 calls). If you see a 'Resource Exhausted' error, it means the AI is tired of your sh*t. Let it sleep it off and try again tomorrow.</p>
          <p className="disclaimer-click-note">click me if you're brave enough</p>
        </div>
      )}
      <p>Got a photo that looks like it went through a blender? We can help. Probably. Upload your digital disaster and let's see what fresh hell we can create.</p>
      
      <div className="file-uploader">
        <label htmlFor="file-upload" className="button">
          {selectedFile ? `Selected: ${selectedFile.name}` : 'Upload Your Digital Disaster'}
        </label>
        <input id="file-upload" type="file" accept="image/*" onChange={handleFileChange} />
      </div>

      <div className="genesis-prompt-container">
        <PromptViewer prompt={GENESIS_PROMPT} title="Show The Batsh*t Crazy Origin Prompt" />
      </div>
      <HowItWorks />
    </div>
  );

  const renderLoading = (customMessage: string) => (
    <div className="app-container" style={panelStyle}>
      <h2 style={titleStyle}>The AI is Cooking... God Knows What</h2>
      <div className="loading-container">
        <div className="spinner"></div>
        <p id="loading-message">{loadingMessage}</p>
        <p className="progress-message">{customMessage}</p>
      </div>
    </div>
  );

  const renderWorkspace = () => {
    if (!originalImage) return null;
    
    const existingTitles = new Set(restorationResults.map(r => r.title));
    const isProcessingCloseups = processingTitles.includes(FACE_CLOSEUP_TITLE);
    const canTransplant = existingTitles.has("Ultimate Quality Remaster") && faceCloseups.length > 0;
    const transplantDone = existingTitles.has("Unholy Face Transplant");
    
    // Logic for ordering and setting the correct 'before' image for transplant
    const ultimateQualityRemaster = restorationResults.find(r => r.title === "Ultimate Quality Remaster");
    const transplantResult = restorationResults.find(r => r.title === "Unholy Face Transplant");
    const otherResults = restorationResults.filter(r => r.title !== "Unholy Face Transplant");

    return (
      <div className="workspace-container">
        <div className="app-container" style={panelStyle}>
          <div className="workspace-header">
              <div className="image-preview-container">
                  <h3 style={titleStyle}>The 'Before' Shot:</h3>
                  <img 
                      src={`data:${originalMimeType};base64,${originalImage}`} 
                      alt="Selected preview" 
                      className="image-preview"
                      onLoad={(e) => !colorPalette && extractColorsFromImage(e.currentTarget).then(setColorPalette)}
                      crossOrigin="anonymous"
                  />
              </div>
              <div className="restoration-controls">
                <h3 style={titleStyle}>The Laboratory</h3>
                <p>Pick your poison. Generate one, or go nuts and generate them all. Click "Iterate" to use a result as the new input for that same style.</p>
                <div className="controls-list">
                  {MASTER_RESTORATION_PROMPTS.map(item => {
                    const isProcessingThisItem = processingTitles.includes(item.title);
                    const isDone = existingTitles.has(item.title);
                    return (
                      <div className="control-item" key={item.title}>
                        <span className="control-item-title">{item.title}</span>
                        <div className="control-buttons-wrapper">
                            <button 
                                className="button-small"
                                onClick={() => handleSingleRestoration(item)}
                                disabled={isProcessingThisItem || isDone}
                            >
                                {isProcessingThisItem && !isDone ? 'Working...' : 'Generate'}
                            </button>
                            {isDone && (
                                <button
                                    className="button-small secondary-button"
                                    onClick={() => handleIterateRestoration(item.title)}
                                    disabled={isProcessingThisItem}
                                >
                                    {isProcessingThisItem ? 'Iterating...' : 'Iterate'}
                                </button>
                            )}
                        </div>
                      </div>
                    )
                  })}
                   <div className="control-item" key={FACE_CLOSEUP_TITLE}>
                        <span className="control-item-title">{FACE_CLOSEUP_TITLE}</span>
                        <button
                            className="button-small"
                            onClick={handleGenerateFaceCloseups}
                            disabled={isProcessingCloseups || faceCloseups.length > 0}
                        >
                            {isProcessingCloseups ? 'Detecting...' : 'Generate'}
                        </button>
                    </div>
                </div>
                <button 
                    className="button secondary-button" 
                    onClick={handleGenerateAll}
                    disabled={processingTitles.length > 0 || existingTitles.size === MASTER_RESTORATION_PROMPTS.length}
                >
                    Generate All Missing
                </button>
              </div>
          </div>
        </div>

        {restorationResults.length > 0 && (
          <div className="results-header app-container" style={panelStyle}>
              <h1 style={titleStyle}>Behold! The Glorious Aftermath</h1>
                <p>Here they are. Your attempts at fixing the mess. Some might be good. Some might be nightmare fuel. You're welcome.</p>
              <div className="transplant-controls">
                {canTransplant && !transplantDone && (
                    <button 
                        className="button" 
                        onClick={() => handleFaceTransplant(false)}
                        disabled={isTransplanting}
                    >
                        {isTransplanting ? "Transplanting..." : "Perform Unholy Face Transplant"}
                    </button>
                )}
                {transplantDone && (
                    <button 
                        className="button" 
                        onClick={() => handleFaceTransplant(true)}
                        disabled={isTransplanting}
                    >
                        {isTransplanting ? "Iterating..." : "Iterate on Transplant"}
                    </button>
                )}
                {isTransplanting && <p className="progress-message">{loadingMessage}</p>}
              </div>
          </div>
        )}
        
        {errorMessage && <p className="error-message" style={{marginTop: '1rem'}}>{errorMessage}</p>}

        <div className="results-grid">
            {otherResults.map((result, index) => (
                <ResultRow
                    key={index}
                    result={result}
                    originalImage={originalImage}
                    originalMimeType={originalMimeType}
                />
            ))}
            {transplantResult && (
                <ResultRow
                    key="transplant"
                    result={transplantResult}
                    originalImage={ultimateQualityRemaster?.data || originalImage}
                    originalMimeType={ultimateQualityRemaster ? 'image/png' : originalMimeType}
                />
            )}
        </div>


        {faceCloseups.length > 0 && (
            <div className="face-closeups-container app-container" style={panelStyle}>
                 <ExpandableSection title={FACE_CLOSEUP_TITLE} defaultOpen={true} level={1}>
                    <p>The AI found {faceCloseups.length} face(s). Compare the results below. Regenerate any you don't like before performing the transplant.</p>
                    <div className="face-closeups-grid">
                        {faceCloseups.map(face => (
                            <div key={face.id} className="face-closeup-item">
                                <h4>{face.title}</h4>
                                <FaceComparisonSlider 
                                    original={face.originalCroppedData} 
                                    restored={face.data} 
                                />
                                <div className="face-closeup-actions">
                                    <button
                                        className="button-small"
                                        onClick={() => handleRegenerateFace(face)}
                                        disabled={processingFaceIds.includes(face.id)}
                                    >
                                        {processingFaceIds.includes(face.id) ? 'Working...' : 'Regenerate'}
                                    </button>
                                     <button 
                                        onClick={() => downloadImage(face.data, generateDownloadFilename(face))} 
                                        className="button-small"
                                    >
                                        Download
                                    </button>
                                </div>
                                <PromptViewer prompt={face.prompt} title="Show Face Prompt" />
                            </div>
                        ))}
                    </div>
                </ExpandableSection>
            </div>
        )}

        <div ref={resultsEndRef} />

        {restorationResults.length > 0 && (
          <div className="app-container" style={panelStyle}>
              <div className="button-group">
                  <button className="button" onClick={handleReset}>Feed Me Another Wreck</button>
              </div>
          </div>
        )}
      </div>
    );
  };
  
    const renderError = () => (
        <div className="app-container" style={panelStyle}>
            <p className="error-message">{errorMessage}</p>
            <button className="button" onClick={handleReset}>I Want To Try Another Abomination</button>
        </div>
    );

    const renderContent = () => {
        switch (appState) {
            case 'idle':
                return originalImage ? renderWorkspace() : renderUploader();
            case 'resuming':
                return renderLoading('Resuming your nonsense...');
            case 'error':
                 return renderError();
            default:
                return (
                    <div className="app-container" style={panelStyle}>
                        <p className="error-message">Something went so wrong, even we don't know what happened. The AI probably unionized.</p>
                        <button className="button" onClick={handleReset}>Start Over</button>
                    </div>
                );
        }
    };

  const isAnythingProcessing = processingTitles.length > 0 || isTransplanting || processingFaceIds.length > 0;

  return (
    <>
      <WavyBackground colorPalette={colorPalette ? colorPalette[theme] : null} />
      <TopBar 
        isProcessing={isAnythingProcessing}
        hasResults={!!originalImage}
        hasSavedSession={hasSavedSession}
        onSave={handleSave} 
        onLoad={handleLoad} 
        onDownloadAll={handleDownloadAll} 
        theme={theme}
        onToggleTheme={toggleTheme}
        isTestMode={isTestMode}
        onToggleTestMode={toggleTestMode}
      />
      <ApiRequestTracker log={apiRequestLog} />
      <main className="main-content-wrapper">
        {renderContent()}
      </main>
    </>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root') as HTMLElement);
root.render(<App />);