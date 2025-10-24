import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import json
import time
from urllib.parse import quote_plus, urljoin
import re
from collections import defaultdict
import os
from dotenv import load_dotenv

class ElonMuskScraperV5:
    """
    Fixed scraper with better date handling and alternative search methods
    """
    
    def __init__(self, newsapi_key=None, output_file='elon_musk_knowledge_base.json'):
        self.newsapi_key = newsapi_key
        self.output_file = output_file
        self.documents = []
        self.existing_urls = set()
        self.existing_content_hashes = set()
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Statistics for debugging
        self.stats = {
            'attempted': 0,
            'no_date': 0,
            'too_short': 0,
            'duplicate': 0,
            'success': 0,
            'errors': 0
        }
        
        self.load_existing()
        
    def load_existing(self):
        """Load existing documents to prevent duplicates"""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                
                for doc in existing:
                    url = doc.get('metadata', {}).get('url', '')
                    if url:
                        self.existing_urls.add(url)
                    
                    content = doc.get('content', '')[:500]
                    self.existing_content_hashes.add(hash(content))
                
                print(f"📂 Loaded {len(existing)} existing documents")
                print(f"   Will skip {len(self.existing_urls)} known URLs\n")
                
            except Exception as e:
                print(f"⚠️  Could not load existing file: {e}\n")
    
    def is_duplicate(self, doc):
        """Check if document is duplicate"""
        url = doc.get('url', '')
        if url and url in self.existing_urls:
            return True
        
        content = doc.get('content', '')[:500]
        if hash(content) in self.existing_content_hashes:
            return True
        
        return False
    
    def get_domain(self, url):
        """Extract domain"""
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        return domain.replace('www.', '')
    
    def clean_text(self, text):
        """Clean text"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
        return text.strip()
    
    def extract_date(self, soup, url="", query_year=None):
        """
        Enhanced date extraction with fallback strategies
        Returns tuple: (date_string, precision)
        - date_string: YYYY-MM-DD format
        - precision: 'exact', 'year_only', or None
        """
        # Strategy 1: time tags
        time_tag = soup.find('time', datetime=True)
        if time_tag and time_tag.get('datetime'):
            try:
                dt_str = time_tag['datetime'].replace('Z', '+00:00')
                if 'T' in dt_str:
                    dt_str = dt_str.split('T')[0]
                dt = datetime.fromisoformat(dt_str)
                return dt.strftime('%Y-%m-%d'), 'exact'
            except:
                pass
        
        # Strategy 2: meta tags
        meta_tags = [
            ('property', 'article:published_time'),
            ('name', 'date'),
            ('property', 'og:published_time'),
            ('name', 'publication_date'),
            ('name', 'publishdate'),
            ('property', 'article:published'),
        ]
        
        for attr, value in meta_tags:
            meta = soup.find('meta', {attr: value})
            if meta and meta.get('content'):
                try:
                    content = meta['content'].replace('Z', '+00:00')
                    if 'T' in content:
                        content = content.split('T')[0]
                    dt = datetime.fromisoformat(content)
                    return dt.strftime('%Y-%m-%d'), 'exact'
                except:
                    pass
        
        # Strategy 3: URL patterns
        url_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', url)
        if url_match:
            return f"{url_match.group(1)}-{url_match.group(2)}-{url_match.group(3)}", 'exact'
        
        url_match = re.search(r'/(\d{4})-(\d{2})-(\d{2})', url)
        if url_match:
            return f"{url_match.group(1)}-{url_match.group(2)}-{url_match.group(3)}", 'exact'
        
        # Strategy 4: Text patterns (more comprehensive)
        text = soup.get_text()[:2000]
        
        # Try full date patterns first
        patterns = [
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})',
            r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',
            r'(\d{4})[-/](\d{2})[-/](\d{2})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    groups = match.groups()
                    if groups[0] in ['January', 'February', 'March', 'April', 'May', 'June', 
                                    'July', 'August', 'September', 'October', 'November', 'December']:
                        dt = datetime.strptime(f"{groups[0]} {groups[1]} {groups[2]}", "%B %d %Y")
                    elif groups[1] in ['January', 'February', 'March', 'April', 'May', 'June', 
                                      'July', 'August', 'September', 'October', 'November', 'December']:
                        dt = datetime.strptime(f"{groups[1]} {groups[0]} {groups[2]}", "%B %d %Y")
                    else:
                        dt = datetime(int(groups[0]), int(groups[1]), int(groups[2]))
                    
                    if 2010 <= dt.year <= datetime.now().year:
                        return dt.strftime('%Y-%m-%d'), 'exact'
                except:
                    pass
        
        # Strategy 5: Fallback to query year if provided
        if query_year:
            # Use middle of the year as estimate
            return f"{query_year}-06-15", 'year_only'
        
        # Strategy 6: Look for just year in URL
        url_year_match = re.search(r'/(\d{4})/', url)
        if url_year_match:
            year = int(url_year_match.group(1))
            if 2010 <= year <= datetime.now().year:
                return f"{year}-06-15", 'year_only'
        
        return None, None
    
    def scrape_article(self, url, source=None, topic=None, query_year=None):
        """Scrape single article with enhanced debugging"""
        self.stats['attempted'] += 1
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for tag in soup(['script', 'style', 'nav', 'footer', 'aside', 'iframe', 'header']):
                tag.decompose()
            
            title = ''
            title_tag = soup.find('h1') or soup.find('title')
            if title_tag:
                title = self.clean_text(title_tag.get_text())
            
            date_str, date_precision = self.extract_date(soup, url, query_year)
            if not date_str:
                self.stats['no_date'] += 1
                return None
            
            article = soup.find('article') or soup.find('main') or soup.find('div', class_=re.compile('article|content|post-content|entry-content|story-body'))
            if article:
                paragraphs = article.find_all('p')
            else:
                paragraphs = soup.find_all('p')
            
            # Remove duplicate paragraphs
            seen_paragraphs = set()
            unique_paragraphs = []
            for p in paragraphs:
                text = p.get_text().strip()
                if len(text) > 30 and text not in seen_paragraphs:
                    seen_paragraphs.add(text)
                    unique_paragraphs.append(text)
            
            content = ' '.join(unique_paragraphs)
            content = self.clean_text(content)
            
            if len(content) < 200:
                self.stats['too_short'] += 1
                return None
            
            if not source:
                source = self.get_domain(url)
            
            self.stats['success'] += 1
            
            return {
                'content': content,
                'date': date_str,
                # 'date_precision': date_precision,
                'source': source,
                'topic': topic or 'General',
                'title': title,
                'url': url
            }
            
        except Exception as e:
            self.stats['errors'] += 1
            return None
    
    def newsapi_search(self, query, topic):
        """NewsAPI - limited to 1 month on free tier"""
        if not self.newsapi_key:
            return []
        
        print(f"  NewsAPI: {query}")
        
        try:
            from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            url = f"https://newsapi.org/v2/everything?q={quote_plus(query)}&from={from_date}&language=en&sortBy=relevancy&pageSize=100&apiKey={self.newsapi_key}"
            
            response = requests.get(url)
            data = response.json()
            
            if data.get('status') != 'ok':
                print(f"    ✗ Error: {data.get('message', 'Unknown')}")
                return []
            
            docs = []
            for article in data.get('articles', []):
                pub_date = article.get('publishedAt', '')
                if pub_date:
                    try:
                        dt = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                        date_str = dt.strftime('%Y-%m-%d')
                    except:
                        continue
                else:
                    continue
                
                description = article.get('description') or ''
                article_content = article.get('content') or ''
                content = description + ' ' + article_content
                content = self.clean_text(content)
                
                if len(content) < 100:
                    continue
                
                article_url = article.get('url', '')
                if not article_url:
                    continue
                
                docs.append({
                    'content': content,
                    'date': date_str,
                    # 'date_precision': 'exact',  # NewsAPI always provides exact dates
                    'source': self.get_domain(article_url),
                    'topic': topic,
                    'title': article.get('title', ''),
                    'url': article_url
                })
            
            print(f"    ✓ {len(docs)} articles")
            return docs
            
        except Exception as e:
            print(f"    ✗ Error: {str(e)}")
            return []
    
    def duckduckgo_search(self, query, topic, query_year=None, max_results=15):
        """
        DuckDuckGo lite search (more reliable endpoint)
        """
        print(f"  DuckDuckGo: {query}")
        
        all_docs = []
        
        try:
            # Use DuckDuckGo lite version which is more stable for scraping
            search_url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"
            response = self.session.get(search_url, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all result links in lite version
            links = []
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                # DuckDuckGo lite uses redirect links
                if href.startswith('http') and 'duckduckgo.com' not in href:
                    links.append(href)
            
            # Remove duplicates
            links = list(set(links))
            
            if not links:
                print(f"    ⚠️  No search results found")
                return []
            
            print(f"    - Found {len(links)} search results")
            
            attempted = 0
            for article_url in links[:30]:  # Try first 30 results
                # Skip social media and video sites
                skip = ['youtube.com', 'facebook.com', 'twitter.com', 'reddit.com', 
                       'tiktok.com', 'instagram.com', 'pinterest.com', 'linkedin.com']
                if any(d in article_url.lower() for d in skip):
                    continue
                
                attempted += 1
                doc = self.scrape_article(article_url, topic=topic, query_year=query_year)
                if doc:
                    if not self.is_duplicate(doc):
                        all_docs.append(doc)
                        self.existing_urls.add(doc['url'])
                        self.existing_content_hashes.add(hash(doc['content'][:500]))
                    else:
                        self.stats['duplicate'] += 1
                
                if len(all_docs) >= max_results:
                    break
                
                time.sleep(0.5)
            
            print(f"    - Attempted {attempted}, extracted {len(all_docs)}")
            print(f"    ✓ {len(all_docs)} new articles")
            return all_docs
            
        except Exception as e:
            print(f"    ✗ Error: {str(e)}")
            return []
    
    def google_search_alternative(self, query, topic, query_year=None, max_results=15):
        """
        Alternative search using a different approach - scrape news aggregator sites directly
        """
        print(f"  Archive Search: {query}")
        
        all_docs = []
        
        # Use news aggregator sites that have good archives
        sources = [
            (f"https://www.theverge.com/search?q={quote_plus(query)}", 'theverge.com'),
            (f"https://techcrunch.com/?s={quote_plus(query)}", 'techcrunch.com'),
            (f"https://arstechnica.com/?s={quote_plus(query)}", 'arstechnica.com'),
        ]
        
        for search_url, source in sources:
            try:
                response = self.session.get(search_url, timeout=15)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find article links
                links = set()
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    if href.startswith('/'):
                        href = f"https://{source}{href}"
                    
                    if source in href and any(x in href for x in ['/20', 'article', query.split()[0].lower()]):
                        links.add(href)
                
                if links:
                    print(f"    - {source}: Found {len(links)} links")
                
                for article_url in list(links)[:10]:
                    doc = self.scrape_article(article_url, source, topic, query_year)
                    if doc and not self.is_duplicate(doc):
                        all_docs.append(doc)
                        self.existing_urls.add(doc['url'])
                        self.existing_content_hashes.add(hash(doc['content'][:500]))
                    
                    if len(all_docs) >= max_results:
                        break
                    
                    time.sleep(0.3)
                
                if len(all_docs) >= max_results:
                    break
                    
            except Exception as e:
                continue
        
        print(f"    ✓ {len(all_docs)} new articles")
        return all_docs
    
    def scrape_specific_sites(self):
        """Scrape specific high-quality sites directly with updated selectors"""
        print("\n📰 Direct Site Scraping:")
        
        # Teslarati - Try multiple entry points
        print(f"  teslarati.com")
        try:
            teslarati_urls = [
                'https://www.teslarati.com/category/spacex/',
                'https://www.teslarati.com/category/tesla/',
                'https://www.teslarati.com/',  # Homepage
            ]
            
            all_links = set()
            for url in teslarati_urls:
                try:
                    response = self.session.get(url, timeout=15)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Look for article links more broadly
                    for a in soup.find_all('a', href=True):
                        href = a['href']
                        if not href.startswith('http'):
                            href = urljoin(url, href)
                        
                        # Match Teslarati article patterns
                        if 'teslarati.com' in href and any(x in href for x in ['/2024/', '/2025/', '/202']):
                            all_links.add(href)
                    
                    time.sleep(1)
                except:
                    continue
            
            print(f"    - Found {len(all_links)} potential links")
            
            count = 0
            attempted = 0
            for article_url in list(all_links)[:30]:
                attempted += 1
                doc = self.scrape_article(article_url, 'teslarati.com', 'Tesla/SpaceX')
                if doc and not self.is_duplicate(doc):
                    self.documents.append(doc)
                    self.existing_urls.add(doc['url'])
                    self.existing_content_hashes.add(hash(doc['content'][:500]))
                    count += 1
                
                if attempted % 10 == 0:
                    print(f"    - Progress: {attempted} attempted, {count} extracted")
                
                time.sleep(0.5)
            
            print(f"    ✓ {count} articles from {attempted} attempts")
            
        except Exception as e:
            print(f"    ✗ Error: {str(e)}")
        
        # Space.com - Better approach
        print(f"  space.com")
        try:
            space_urls = [
                'https://www.space.com/spacex',
                'https://www.space.com/topics/commercial-spaceflight',
                'https://www.space.com/news',
            ]
            
            all_links = set()
            for url in space_urls:
                try:
                    response = self.session.get(url, timeout=15)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    for a in soup.find_all('a', href=True):
                        href = a['href']
                        if href.startswith('/'):
                            href = f"https://www.space.com{href}"
                        
                        # Match Space.com article patterns
                        if 'space.com' in href and not any(x in href for x in ['#', 'search', 'topics', 'tag']):
                            # Check if it looks like an article (has hyphens, not just category pages)
                            if href.count('/') >= 3 and '-' in href:
                                all_links.add(href)
                    
                    time.sleep(1)
                except:
                    continue
            
            print(f"    - Found {len(all_links)} potential links")
            
            count = 0
            attempted = 0
            for article_url in list(all_links)[:30]:
                attempted += 1
                doc = self.scrape_article(article_url, 'space.com', 'SpaceX')
                if doc and not self.is_duplicate(doc):
                    self.documents.append(doc)
                    self.existing_urls.add(doc['url'])
                    self.existing_content_hashes.add(hash(doc['content'][:500]))
                    count += 1
                
                if attempted % 10 == 0:
                    print(f"    - Progress: {attempted} attempted, {count} extracted")
                
                time.sleep(0.5)
            
            print(f"    ✓ {count} articles from {attempted} attempts")
            
        except Exception as e:
            print(f"    ✗ Error: {str(e)}")
        
        # Electrek - Keep existing approach (it's working)
        print(f"  electrek.co")
        try:
            url = 'https://electrek.co/guides/elon-musk/'
            response = self.session.get(url, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            links = set()
            for a in soup.find_all('a', href=True):
                href = a['href']
                if href.startswith('/'):
                    href = f"https://electrek.co{href}"
                
                if 'electrek.co' in href and any(x in href for x in ['/20', 'article', 'post']):
                    links.add(href)
            
            print(f"    - Found {len(links)} potential links")
            
            count = 0
            attempted = 0
            for article_url in list(links)[:30]:
                attempted += 1
                doc = self.scrape_article(article_url, 'electrek.co', 'Tesla/EV')
                if doc and not self.is_duplicate(doc):
                    self.documents.append(doc)
                    self.existing_urls.add(doc['url'])
                    self.existing_content_hashes.add(hash(doc['content'][:500]))
                    count += 1
                
                if attempted % 10 == 0:
                    print(f"    - Progress: {attempted} attempted, {count} extracted")
                
                time.sleep(0.5)
            
            print(f"    ✓ {count} articles from {attempted} attempts")
            
        except Exception as e:
            print(f"    ✗ Error: {str(e)}")
    
    def run(self):
        """Main execution"""
        print("=" * 70)
        print("ELON MUSK KNOWLEDGE BASE SCRAPER v5.0 - FIXED VERSION")
        print("=" * 70)
        
        # Phase 1: Recent news
        if self.newsapi_key:
            print("\n📰 Phase 1: Recent News (Last 30 days)")
            
            queries = [
                ('elon musk tesla', 'Tesla'),
                ('elon musk spacex', 'SpaceX'),
                ('elon musk neuralink', 'Neuralink'),
                ('elon musk xai grok', 'xAI/Grok'),
                ('elon musk trump', 'Politics'),
                ('elon musk DOGE', 'DOGE'),
                ('elon musk starlink', 'Starlink'),
            ]
            
            for query, topic in queries:
                docs = self.newsapi_search(query, topic)
                new_docs = [d for d in docs if not self.is_duplicate(d)]
                self.documents.extend(new_docs)
                for d in new_docs:
                    self.existing_urls.add(d['url'])
                    self.existing_content_hashes.add(hash(d['content'][:500]))
                time.sleep(1)
        
        # Phase 2: Historical content with multiple strategies
        print("\n🔍 Phase 2: Historical Content (2015-2024)")
        
        historical_searches = [
            ('tesla IPO 2010 elon musk', 'Tesla Historical', 2010),
            ('spacex falcon 9 landing 2015', 'SpaceX Historical', 2015),
            ('tesla model 3 reveal 2016', 'Tesla Historical', 2016),
            ('spacex falcon heavy 2018', 'SpaceX Historical', 2018),
            ('elon musk twitter acquisition 2022', 'Twitter/X', 2022),
            ('neuralink human trial 2024', 'Neuralink', 2024),
            ('xai grok chatbot 2023', 'xAI/Grok', 2023),
            ('tesla gigafactory berlin 2022', 'Tesla', 2022),
            ('spacex crew dragon 2020', 'SpaceX', 2020),
            ('tesla cybertruck 2023', 'Tesla', 2023),
            ('spacex starship flight', 'SpaceX', 2023),
        ]
        
        for query, topic, year in historical_searches:
            # Try DuckDuckGo first
            docs = self.duckduckgo_search(query, topic, query_year=year, max_results=10)
            
            # If DuckDuckGo fails, try alternative search
            if len(docs) == 0:
                docs = self.google_search_alternative(query, topic, query_year=year, max_results=10)
            
            self.documents.extend(docs)
            time.sleep(2)  # Be respectful
        
        # Phase 3: Direct site scraping
        self.scrape_specific_sites()
        
        # Statistics
        print("\n" + "=" * 70)
        print(f"📊 Collected {len(self.documents)} NEW documents this run")
        print(f"\n🔍 Scraping Statistics:")
        print(f"   Attempted: {self.stats['attempted']}")
        print(f"   Success: {self.stats['success']}")
        print(f"   No date found: {self.stats['no_date']}")
        print(f"   Content too short: {self.stats['too_short']}")
        print(f"   Duplicates: {self.stats['duplicate']}")
        print(f"   Errors: {self.stats['errors']}")
        
        if len(self.documents) == 0:
            print("\n⚠️  No new documents found")
            return self.documents
        
        topic_counts = defaultdict(int)
        year_counts = defaultdict(int)
        # precision_counts = defaultdict(int)
        
        for doc in self.documents:
            topic_counts[doc.get('topic', 'Unknown')] += 1
            date = doc.get('date', '')
            if date and len(date) >= 4:
                year = date[:4]
                year_counts[year] += 1
            # precision_counts[doc.get('date_precision', 'unknown')] += 1
        
        print("\n📂 New Documents by Topic:")
        for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {topic}: {count}")
        
        print("\n📅 New Documents by Year:")
        for year, count in sorted(year_counts.items(), reverse=True):
            print(f"  {year}: {count}")
        
        # print("\n🎯 Date Precision:")
        # print(f"  Exact dates: {precision_counts.get('exact', 0)}")
        # print(f"  Year-only estimates: {precision_counts.get('year_only', 0)}")
        
        return self.documents
    
    def save(self):
        """Save all documents"""
        all_docs = []
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r', encoding='utf-8') as f:
                all_docs = json.load(f)
        
        existing_count = len(all_docs)
        
        for doc in self.documents:
            if not doc.get('date') or not doc.get('source') or len(doc.get('content', '')) < 200:
                continue
            
            all_docs.append({
                'content': doc['content'],
                'date': doc['date'],
                'source': doc['source'],
                'metadata': {
                    'title': doc.get('title', ''),
                    'topic': doc.get('topic', ''),
                    'url': doc.get('url', ''),
                    # 'date_precision': doc.get('date_precision', 'unknown')
                }
            })
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(all_docs, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 70)
        print(f"✅ Saved to {self.output_file}")
        print(f"   Total documents: {len(all_docs)}")
        print(f"   Existing: {existing_count}")
        print(f"   Newly added: {len(all_docs) - existing_count}")
        print("=" * 70)


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    API_KEY = os.getenv("NEWSAPI_KEY")
    
    if not API_KEY:
        print("⚠️  Warning: NEWSAPI_KEY not found in .env file")
        print("   Some features will be disabled")
    
    scraper = ElonMuskScraperV5(newsapi_key=API_KEY)
    scraper.run()
    scraper.save()
    
    print("\n✅ Done! Run again anytime to add more content.")