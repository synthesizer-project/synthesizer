/**
 * Google Analytics (GA4) Configuration
 * ------------------------------------------------------------------
 * This script initializes the Global Site Tag (gtag.js) required for 
 * sending data to Google Analytics.
 *
 * It is loaded by Sphinx via the `html_js_files` configuration in conf.py.
 */

// Initialize the data layer array if it doesn't exist.
// This is where GA buffers events before processing them.
window.dataLayer = window.dataLayer || [];
// Define the gtag function which pushes arguments to the data layer.
function gtag(){
    dataLayer.push(arguments);
}
// Record the current time to timestamp the session start
gtag("js", new Date());

// Configure the specific Google Analytics property ID.
gtag("config", "G-CKBVJ4RR3N");