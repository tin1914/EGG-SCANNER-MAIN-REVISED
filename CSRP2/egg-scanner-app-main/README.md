# Fresh Eggs Scanner Website

A complete HTML/CSS web application for scanning and assessing egg quality, featuring both individual and batch scanning workflows.

## Features

- **Homepage** - Landing page with hero section, process steps, and guidelines
- **Individual Egg Scanning** - Upload single egg images for quality assessment
- **Batch Egg Scanning** - Upload multiple egg images for batch processing
- **Loading Pages** - 5-second animated loading states with automatic transitions
- **Results Pages** - Display quality assessment results (FRESH status with percentages)
- **Responsive Design** - Works on desktop, tablet, and mobile devices
- **Smooth Animations** - CSS animations for loading states and interactions

## Page Flow

### Individual Egg Flow:
1. `index.html` → Homepage
2. `individual-upload.html` → Single egg upload
3. `individual-loading.html` → Loading (5 seconds)
4. `individual-results.html` → Results display

### Batch Egg Flow:
1. `index.html` → Homepage  
2. `batch-upload.html` → Multiple egg upload
3. `batch-loading.html` → Loading (5 seconds)
4. `batch-results.html` → Batch results display

## File Structure

```
/
├── index.html                 # Homepage
├── individual-upload.html     # Single egg upload
├── individual-loading.html    # Individual loading page
├── individual-results.html    # Individual results
├── batch-upload.html         # Batch upload page
├── batch-loading.html        # Batch loading page
├── batch-results.html        # Batch results
├── styles/
│   └── main.css              # All styling
└── assets/
    └── images/               # Image assets directory
```

## How to Use

1. Open `index.html` in a web browser
2. Click "SCAN YOUR EGGS HERE" to start individual scanning
3. Or click "SCAN ANOTHER SET OF EGGS!" for batch scanning
4. Use "BROWSE FILE" to select images (functionality simulated)
5. Click scan buttons to proceed through the flow
6. Loading pages automatically redirect after 5 seconds
7. Results pages show quality assessment with "SCAN ANOTHER EGG!" option

## Navigation

- **Home** button - Returns to homepage
- **Contact** button - Contact section (placeholder)
- All scan buttons navigate through the appropriate flow
- Automatic redirects on loading pages

## Styling

- Yellow (#FFC107) - Primary color for process steps and headers
- Green (#4CAF50) - Success states and fresh indicators  
- Red (#F44336) - Action buttons and progress bars
- Responsive design with mobile-first approach
- Smooth hover effects and transitions
- CSS animations for loading states

## Browser Compatibility

Works in all modern browsers including:
- Chrome
- Firefox  
- Safari
- Edge

No JavaScript frameworks required - uses only HTML, CSS, and minimal JavaScript for page redirects.