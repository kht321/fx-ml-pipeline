// News Simulator Frontend JavaScript

// API base URL
const API_BASE = window.location.origin;

// Update statistics on page load
window.addEventListener('DOMContentLoaded', () => {
    updateStats();
    // Refresh stats every 5 seconds
    setInterval(updateStats, 5000);
});

/**
 * Fetch and update statistics from backend
 */
async function updateStats() {
    try {
        const response = await fetch(`${API_BASE}/api/stats`);
        const data = await response.json();

        // Update total articles
        document.getElementById('total-articles').textContent = data.total_articles.toLocaleString();
        document.getElementById('streamed-count').textContent = data.streamed_count.toLocaleString();

        // Update breakdown
        document.getElementById('positive-count').textContent = data.positive.toLocaleString();
        document.getElementById('negative-count').textContent = data.negative.toLocaleString();
        document.getElementById('neutral-count').textContent = data.neutral.toLocaleString();

        // Update last article if available
        if (data.last_article) {
            displayLastArticle(data.last_article);
        }

        // Update status
        setStatus('ready', 'Ready');

    } catch (error) {
        console.error('Error updating stats:', error);
        setStatus('error', 'Connection Error');
    }
}

/**
 * Stream news article of specified sentiment type
 * @param {string} sentimentType - "positive", "negative", or "neutral"
 */
async function streamNews(sentimentType) {
    const buttons = document.querySelectorAll('.btn');
    const clickedButton = event.target.closest('.btn');

    // Disable all buttons and show loading
    buttons.forEach(btn => btn.classList.add('loading'));
    setStatus('loading', 'Streaming...');

    try {
        const response = await fetch(`${API_BASE}/api/stream/${sentimentType}`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.status === 'success') {
            // Display success
            displayLastArticle({
                time: data.article.streamed_at,
                headline: data.article.headline,
                sentiment: data.article.sentiment_score,
                type: sentimentType
            });

            // Show success animation
            showNotification(`✓ Streamed ${sentimentType} article`, 'success');

            // Update stats
            await updateStats();

        } else {
            throw new Error(data.error || 'Unknown error');
        }

    } catch (error) {
        console.error('Error streaming news:', error);
        showNotification(`✗ Error: ${error.message}`, 'error');
        setStatus('error', 'Error');
    } finally {
        // Re-enable buttons
        buttons.forEach(btn => btn.classList.remove('loading'));
        setTimeout(() => setStatus('ready', 'Ready'), 2000);
    }
}

/**
 * Display the last streamed article
 * @param {Object} article - Article data
 */
function displayLastArticle(article) {
    const container = document.getElementById('last-article');
    const time = document.getElementById('article-time');
    const headline = document.getElementById('article-headline');
    const badge = document.getElementById('article-sentiment-badge');
    const score = document.getElementById('article-sentiment-score');

    // Format time
    const date = new Date(article.time);
    time.textContent = date.toLocaleString();

    // Set headline
    headline.textContent = article.headline;

    // Set sentiment badge
    badge.className = 'badge';
    if (article.type === 'positive') {
        badge.classList.add('badge-positive');
        badge.textContent = 'Positive';
    } else if (article.type === 'negative') {
        badge.classList.add('badge-negative');
        badge.textContent = 'Negative';
    } else {
        badge.classList.add('badge-neutral');
        badge.textContent = 'Neutral';
    }

    // Set sentiment score
    const sentimentValue = article.sentiment.toFixed(2);
    score.textContent = `Sentiment: ${sentimentValue}`;

    // Show container
    container.style.display = 'block';

    // Add fade-in animation
    container.style.animation = 'none';
    setTimeout(() => {
        container.style.animation = 'fadeIn 0.5s ease';
    }, 10);
}

/**
 * Set status indicator
 * @param {string} type - "ready", "loading", "error", "warning"
 * @param {string} text - Status text
 */
function setStatus(type, text) {
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.getElementById('status-text');

    statusDot.className = 'status-dot';
    if (type === 'error') {
        statusDot.classList.add('error');
    } else if (type === 'warning') {
        statusDot.classList.add('warning');
    }

    statusText.textContent = text;
}

/**
 * Show temporary notification
 * @param {string} message - Notification message
 * @param {string} type - "success" or "error"
 */
function showNotification(message, type) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 16px 24px;
        background: ${type === 'success' ? '#28a745' : '#dc3545'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        font-weight: 500;
        z-index: 1000;
        animation: slideIn 0.3s ease;
    `;

    document.body.appendChild(notification);

    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

/**
 * Reload articles from disk
 */
async function reloadArticles() {
    const button = event.target;
    button.disabled = true;
    button.textContent = '🔄 Reloading...';

    try {
        const response = await fetch(`${API_BASE}/api/reload`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.status === 'success') {
            showNotification(`✓ Reloaded ${data.total_articles} articles`, 'success');
            await updateStats();
        } else {
            throw new Error(data.error || 'Unknown error');
        }

    } catch (error) {
        console.error('Error reloading articles:', error);
        showNotification(`✗ Error: ${error.message}`, 'error');
    } finally {
        button.disabled = false;
        button.textContent = '🔄 Reload Articles';
    }
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }

    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);
