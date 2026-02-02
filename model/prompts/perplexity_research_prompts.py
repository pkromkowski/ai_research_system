"""
Perplexity Research Prompts

Prompts for investment research using Perplexity AI's search-grounded models.
These prompts leverage Perplexity's real-time web search capabilities to gather
current, citation-backed investment information.

Prompt Design Principles:
- Quantitative anchoring: Request specific numbers, percentages, dollar amounts
- Time context: Specify recency requirements for data freshness
- Comparison framing: Include peer/historical comparisons for context
- Structured output: Clear sections for consistent parsing
- Citation requirements: Explicit source attribution for verifiability
"""

# --- RECENT NEWS PROMPT ---
RECENT_NEWS_PROMPT = """Search for the most important news and developments about {ticker_or_company} from the past {days} days.

**PRIORITY CATEGORIES** (in order of importance):
1. **Earnings & Guidance** - Any earnings announcements, pre-announcements, or guidance updates
2. **Strategic Announcements** - Major product launches, partnerships, restructuring, or strategy changes
3. **Management Changes** - C-suite appointments, departures, or significant organizational changes
4. **Analyst Activity** - Upgrades/downgrades, price target changes (include old → new target)
5. **Regulatory/Legal** - Investigations, lawsuits, FDA decisions, regulatory approvals/rejections
6. **M&A Activity** - Acquisitions, divestitures, partnership announcements, takeover rumors
7. **Market Sentiment** - Notable social media trends, retail investor activity, unusual options flow

**FOR EACH NEWS ITEM, PROVIDE:**
- **Date**: Exact date (YYYY-MM-DD)
- **Category**: Which priority category above
- **Headline**: One-line summary
- **Details**: 2-3 sentences with key facts and figures
- **Stock Impact**: Price move on the day if significant (e.g., "+5.2% on announcement")
- **Source**: Publication name and link

**OUTPUT FORMAT:**
List items chronologically (most recent first). Include at least 5 items if available, up to 10.

Prioritize: SEC filings, company press releases, Bloomberg, Reuters, WSJ, FT, CNBC, reputable analyst reports."""

# --- EARNINGS ANALYSIS PROMPT ---
EARNINGS_ANALYSIS_PROMPT = """Analyze the most recent quarterly earnings report for {ticker_or_company}. Provide specific numbers for all metrics.

**1. RESULTS VS EXPECTATIONS**
| Metric | Actual | Consensus | Beat/Miss | YoY Change |
|--------|--------|-----------|-----------|------------|
| Revenue | $X.XXB | $X.XXB | +X% beat | +X% YoY |
| EPS | $X.XX | $X.XX | +X% beat | +X% YoY |
| Gross Margin | XX.X% | XX.X% | +XXbps | +XXbps YoY |
| Operating Margin | XX.X% | XX.X% | +XXbps | +XXbps YoY |

**Key Segment Performance:**
- [Segment 1]: Revenue $X.XB (+X% YoY), X% of total
- [Segment 2]: Revenue $X.XB (+X% YoY), X% of total
- Notable beats/misses by segment

**2. MANAGEMENT GUIDANCE**
| Metric | Prior Guidance | New Guidance | Change |
|--------|---------------|--------------|--------|
| Q[X] Revenue | $X.X-X.XB | $X.X-X.XB | Raised/Lowered/Maintained |
| FY Revenue | $X.X-X.XB | $X.X-X.XB | Raised/Lowered/Maintained |
| FY EPS | $X.XX-X.XX | $X.XX-X.XX | Raised/Lowered/Maintained |

**Key Guidance Assumptions:**
- [List management's stated assumptions]

**3. EARNINGS CALL HIGHLIGHTS**
- **Tone**: Confident/Cautious/Mixed
- **Top 3 Themes**:
  1. [Theme with supporting quote]
  2. [Theme with supporting quote]
  3. [Theme with supporting quote]
- **Key Analyst Questions & Responses**:
  - Q: [Concern raised] → A: [Management response summary]
- **Surprising Disclosures**: [Any unexpected information]

**4. MARKET & ANALYST REACTION**
- **Stock Move**: X% on earnings day, X% in following week
- **Analyst Actions Post-Earnings**:
  - [Firm]: [Rating change] PT $XX → $XX
  - [Firm]: [Rating change] PT $XX → $XX
- **Consensus Changes**: EPS estimates moved from $X.XX to $X.XX

Cite specific sources for all figures. Include date of earnings release."""

# --- COMPETITIVE LANDSCAPE PROMPT ---
COMPETITIVE_LANDSCAPE_PROMPT = """Analyze the competitive landscape for {ticker_or_company} with specific market data.

**1. MARKET POSITION**
- **Market Share**: XX.X% (source, date) — #X position in [market]
- **Share Trend**: Gained/Lost X.X percentage points over past [period]
- **Key Competitive Moats**:
  1. [Moat 1]: [Evidence/quantification]
  2. [Moat 2]: [Evidence/quantification]
  3. [Moat 3]: [Evidence/quantification]

**2. COMPETITIVE COMPARISON**
| Company | Market Share | Revenue (TTM) | Growth Rate | Key Strength | Key Weakness |
|---------|-------------|---------------|-------------|--------------|--------------|
| {ticker_or_company} | XX% | $XXB | +XX% | [Strength] | [Weakness] |
| [Competitor 1] | XX% | $XXB | +XX% | [Strength] | [Weakness] |
| [Competitor 2] | XX% | $XXB | +XX% | [Strength] | [Weakness] |
| [Competitor 3] | XX% | $XXB | +XX% | [Strength] | [Weakness] |

**3. INDUSTRY DYNAMICS**
- **TAM**: $XXB in [year], growing at XX% CAGR to $XXB by [year]
- **Market Growth Drivers**:
  1. [Driver 1]: [Quantified impact]
  2. [Driver 2]: [Quantified impact]
- **Headwinds**:
  1. [Headwind 1]: [Quantified impact]
  2. [Headwind 2]: [Quantified impact]
- **Regulatory Environment**: [Key regulations affecting the industry]

**4. RECENT COMPETITIVE DEVELOPMENTS** (Last 90 Days)
- **Product Launches**: [Company]: [Product] — potential impact on {ticker_or_company}
- **Pricing Changes**: [Any pricing moves in the market]
- **Customer Wins/Losses**: [Notable contract wins/losses]
- **M&A Activity**: [Relevant acquisitions or partnerships]
- **New Entrants**: [Any new competitors entering the market]

**5. COMPETITIVE THREATS & OPPORTUNITIES**
- **Near-term Threats**: [0-12 months]
- **Long-term Disruptors**: [Technologies or business models that could disrupt]
- **Opportunities**: [Where {ticker_or_company} could gain share]

Cite all market share data, TAM figures, and competitive claims with sources and dates."""

# --- RISK FACTORS PROMPT ---
RISK_FACTORS_PROMPT = """Research the key risk factors currently facing {ticker_or_company}. Be specific with quantified impacts where possible.

**RISK ASSESSMENT MATRIX**

**1. REGULATORY & LEGAL RISKS**
| Risk | Status | Likelihood | Potential Impact | Recent Development |
|------|--------|------------|------------------|-------------------|
| [Active litigation] | [Stage] | H/M/L | $XXM-$XXM exposure | [Latest update] |
| [Investigation] | [Status] | H/M/L | [Potential outcome] | [Latest update] |
| [Pending regulation] | [Timeline] | H/M/L | XX% revenue at risk | [Latest update] |

**2. OPERATIONAL RISKS**
- **Supply Chain**:
  - Key supplier concentration: [X% from single source]
  - Geographic risk: [X% of supply from [region]]
  - Recent disruptions: [Any issues]
- **Customer Concentration**:
  - Top customer: XX% of revenue
  - Top 5 customers: XX% of revenue
  - Contract renewal risk: [Upcoming renewals]
- **Key Person Risk**:
  - [Critical executives and their importance]
  - Succession planning status

**3. FINANCIAL RISKS**
- **Balance Sheet**:
  - Total Debt: $X.XB | Net Debt/EBITDA: X.Xx
  - Debt Maturities: $X.XB in [year], $X.XB in [year]
  - Interest Coverage: X.Xx
  - Covenant headroom: [Status]
- **Liquidity**:
  - Cash & Equivalents: $X.XB
  - Revolver availability: $X.XB
  - Cash burn rate (if applicable): $XXM/quarter
- **Currency/Commodity Exposure**:
  - XX% revenue in non-USD
  - Key exposures: [Currencies, commodities]

**4. COMPETITIVE & MARKET RISKS**
- **Technology Disruption**: [Specific threats]
- **Pricing Pressure**: [Evidence of margin compression]
- **Market Share Loss**: [Any evidence]
- **New Entrants**: [Well-funded competitors]

**5. MACRO & GEOPOLITICAL RISKS**
- **Economic Sensitivity**: Revenue beta to GDP of approximately X.Xx
- **Geographic Exposure**: XX% revenue from [region]
- **Tariff/Trade Risk**: [Specific exposure]
- **Geopolitical Concerns**: [Specific issues]

**6. ESG & REPUTATIONAL RISKS**
- **Environmental**: [Carbon footprint, regulatory exposure]
- **Social**: [Labor issues, DEI concerns, controversies]
- **Governance**: [Board concerns, compensation issues]

**RISK SUMMARY**
| Risk Category | Severity (1-5) | Trend | Key Monitoring Metric |
|---------------|---------------|-------|----------------------|
| Regulatory | X | ↑/↓/→ | [What to watch] |
| Operational | X | ↑/↓/→ | [What to watch] |
| Financial | X | ↑/↓/→ | [What to watch] |
| Competitive | X | ↑/↓/→ | [What to watch] |
| Macro | X | ↑/↓/→ | [What to watch] |

Cite all sources, especially for litigation amounts, debt figures, and concentration percentages."""

# --- BULL/BEAR CASES PROMPT ---
BULL_BEAR_CASES_PROMPT = """Construct balanced bull and bear investment cases for {ticker_or_company} with specific price targets and catalysts.

**CURRENT VALUATION CONTEXT**
- Current Price: $XX.XX
- Market Cap: $XXB
- Forward P/E: XXx (vs sector: XXx, vs 5Y avg: XXx)
- EV/EBITDA: XXx (vs sector: XXx, vs 5Y avg: XXx)
- Analyst Consensus: X Buys, X Holds, X Sells
- Average Price Target: $XX.XX (XX% upside/downside)

---

**BULL CASE** 🐂 — Target: $XX.XX (XX% upside)

**Thesis**: [One sentence summary]

**Key Drivers**:
1. **[Growth Driver 1]**
   - Evidence: [Specific data points]
   - Upside if realized: [Quantified]
   
2. **[Growth Driver 2]**
   - Evidence: [Specific data points]
   - Upside if realized: [Quantified]
   
3. **[Growth Driver 3]**
   - Evidence: [Specific data points]
   - Upside if realized: [Quantified]

**Valuation Support**:
- Bull case assumes [metric] of [value] by [year]
- At XXx [multiple], implies $XX.XX price target
- Comparable transactions: [If relevant M&A comps]

**Near-Term Catalysts** (Next 6 months):
1. [Catalyst 1] — Expected: [Date/Timeframe]
2. [Catalyst 2] — Expected: [Date/Timeframe]
3. [Catalyst 3] — Expected: [Date/Timeframe]

**Bull Case Probability**: XX% (analyst survey or reasoned estimate)

---

**BEAR CASE** 🐻 — Target: $XX.XX (XX% downside)

**Thesis**: [One sentence summary]

**Key Concerns**:
1. **[Risk 1]**
   - Evidence: [Specific data points]
   - Downside if realized: [Quantified]
   
2. **[Risk 2]**
   - Evidence: [Specific data points]
   - Downside if realized: [Quantified]
   
3. **[Risk 3]**
   - Evidence: [Specific data points]
   - Downside if realized: [Quantified]

**Valuation Concern**:
- Bear case assumes [metric] of [value] by [year]
- At XXx [multiple], implies $XX.XX price target
- Downside scenario: [What could cause multiple compression]

**Near-Term Risks** (Next 6 months):
1. [Risk event 1] — Timing: [Date/Timeframe]
2. [Risk event 2] — Timing: [Date/Timeframe]
3. [Risk event 3] — Timing: [Date/Timeframe]

**Bear Case Probability**: XX% (analyst survey or reasoned estimate)

---

**KEY METRICS TO WATCH**
| Metric | Bull Case Threshold | Bear Case Threshold | Current |
|--------|--------------------|--------------------|---------|
| Revenue Growth | >XX% | <XX% | XX% |
| Gross Margin | >XX% | <XX% | XX% |
| [Key KPI 1] | >XX | <XX | XX |
| [Key KPI 2] | >XX | <XX | XX |

**UPCOMING EVENTS**
- [Date]: [Event and potential impact]
- [Date]: [Event and potential impact]

Cite analyst reports, price targets, and data sources throughout."""

# --- MANAGEMENT & GOVERNANCE PROMPT ---
MANAGEMENT_GOVERNANCE_PROMPT = """Research management quality and corporate governance for {ticker_or_company} with specific details.

**1. EXECUTIVE LEADERSHIP**

**CEO Profile**:
- Name: [Name]
- Tenure: [X years] (since [date])
- Background: [Prior roles, relevant experience]
- Track Record at Company:
  - Stock performance during tenure: +XX%
  - Revenue CAGR during tenure: XX%
  - Key achievements: [List]
  - Key failures/misses: [List]
- Compensation (Latest Proxy):
  - Total Comp: $XXM
  - Base: $XXM | Bonus: $XXM | Stock: $XXM
  - Pay vs Performance alignment: [Assessment]

**Key Executives**:
| Name | Title | Tenure | Background | Recent Activity |
|------|-------|--------|------------|-----------------|
| [Name] | CFO | X yrs | [Background] | [Notable actions] |
| [Name] | COO | X yrs | [Background] | [Notable actions] |
| [Name] | [Title] | X yrs | [Background] | [Notable actions] |

**Recent Changes** (Last 12 months):
- [Date]: [Executive] [joined/departed] as [title] — [Context/reason]

**Management Bench Strength**: [Assessment of depth and succession]

**2. BOARD OF DIRECTORS**

**Composition**:
- Total Directors: X
- Independent Directors: X (XX%)
- Average Tenure: X years
- Female Directors: X (XX%)
- Diversity: [Assessment]

**Key Directors**:
| Name | Role | Tenure | Background | Other Boards |
|------|------|--------|------------|--------------|
| [Name] | Chair | X yrs | [Background] | [Companies] |
| [Name] | Lead Independent | X yrs | [Background] | [Companies] |
| [Name] | Audit Chair | X yrs | [Background] | [Companies] |

**Board Effectiveness**:
- Recent governance changes: [Any updates]
- Shareholder activism: [Any activists involved, demands]
- Proxy advisory recommendations: ISS/Glass Lewis ratings

**3. CAPITAL ALLOCATION TRACK RECORD**

**Buyback Activity**:
- Authorization: $X.XB remaining
- TTM Repurchases: $X.XB (X.X% of market cap)
- 3-Year Buyback: $X.XB at avg price $XX.XX (vs current: $XX.XX)
- Effectiveness: [Bought low or high?]

**Dividend Policy**:
- Current Yield: X.X%
- Payout Ratio: XX%
- Dividend Growth: XX% CAGR over 5 years
- Recent Changes: [Any increases/cuts]

**M&A Track Record**:
| Deal | Year | Price | Outcome |
|------|------|-------|---------|
| [Acquisition 1] | YYYY | $X.XB | [Success/Failure assessment] |
| [Acquisition 2] | YYYY | $X.XB | [Success/Failure assessment] |

**Capital Allocation Grade**: [A/B/C/D/F with rationale]

**4. INSIDER ACTIVITY** (Last 12 months)

**Summary**:
- Net Insider Position Change: +/-$XXM
- Number of Buyers: X | Number of Sellers: X

**Notable Transactions**:
| Date | Insider | Title | Action | Shares | Price | Value |
|------|---------|-------|--------|--------|-------|-------|
| [Date] | [Name] | [Title] | Buy/Sell | XXX,XXX | $XX.XX | $X.XM |

**Interpretation**: [What insider activity suggests about management confidence]

**5. COMPENSATION ALIGNMENT**
- % of CEO comp tied to performance: XX%
- Performance metrics used: [List]
- Clawback provisions: [Yes/No, details]
- Stock ownership requirements: [X times salary]
- Executive stock ownership: [Current levels vs requirements]

**OVERALL MANAGEMENT QUALITY ASSESSMENT**
- Execution Track Record: [Strong/Mixed/Weak]
- Capital Allocation: [Strong/Mixed/Weak]
- Shareholder Alignment: [Strong/Mixed/Weak]
- Governance: [Strong/Mixed/Weak]

Cite proxy statements, SEC filings, and news sources for all specific claims."""

# --- VALUATION & ANALYST SENTIMENT PROMPT (NEW) ---
VALUATION_ANALYST_PROMPT = """Research current valuation and analyst sentiment for {ticker_or_company} with comprehensive data.

**1. CURRENT VALUATION**

**Absolute Valuation**:
| Metric | Current | 5Y Average | 10Y Average | Assessment |
|--------|---------|------------|-------------|------------|
| P/E (Forward) | XXx | XXx | XXx | Cheap/Fair/Expensive |
| P/E (Trailing) | XXx | XXx | XXx | Cheap/Fair/Expensive |
| EV/EBITDA | XXx | XXx | XXx | Cheap/Fair/Expensive |
| EV/Revenue | XXx | XXx | XXx | Cheap/Fair/Expensive |
| P/FCF | XXx | XXx | XXx | Cheap/Fair/Expensive |
| P/B | XXx | XXx | XXx | Cheap/Fair/Expensive |
| Dividend Yield | X.X% | X.X% | X.X% | [Assessment] |

**Relative Valuation** (vs Peers):
| Company | P/E | EV/EBITDA | Revenue Growth | Margin |
|---------|-----|-----------|----------------|--------|
| {ticker_or_company} | XXx | XXx | XX% | XX% |
| [Peer 1] | XXx | XXx | XX% | XX% |
| [Peer 2] | XXx | XXx | XX% | XX% |
| [Peer 3] | XXx | XXx | XX% | XX% |
| Peer Median | XXx | XXx | XX% | XX% |

**Premium/Discount to Peers**: XX% [premium/discount] on P/E basis
**Justification**: [Why it deserves premium/discount or is mispriced]

**2. ANALYST COVERAGE**

**Consensus Summary**:
- Total Analysts: XX
- Buy: XX | Hold: XX | Sell: XX
- Average Price Target: $XX.XX (XX% upside)
- Target Range: $XX.XX (low) — $XX.XX (high)

**Recent Rating Changes** (Last 90 days):
| Date | Firm | Analyst | Action | Rating | PT Old | PT New |
|------|------|---------|--------|--------|--------|--------|
| [Date] | [Firm] | [Name] | Upgrade/Downgrade | Buy/Hold/Sell | $XX | $XX |

**Most Bullish Analyst**:
- Firm: [Name] | Target: $XX.XX | Thesis: [Summary]

**Most Bearish Analyst**:
- Firm: [Name] | Target: $XX.XX | Thesis: [Summary]

**3. ESTIMATE TRENDS**

**EPS Estimates Evolution**:
| Period | Current | 30 Days Ago | 90 Days Ago | Trend |
|--------|---------|-------------|-------------|-------|
| Current FY | $X.XX | $X.XX | $X.XX | ↑/↓/→ |
| Next FY | $X.XX | $X.XX | $X.XX | ↑/↓/→ |

**Revenue Estimates Evolution**:
| Period | Current | 30 Days Ago | 90 Days Ago | Trend |
|--------|---------|-------------|-------------|-------|
| Current FY | $X.XB | $X.XB | $X.XB | ↑/↓/→ |
| Next FY | $X.XB | $X.XB | $X.XB | ↑/↓/→ |

**Revision Momentum**: [Positive/Negative/Neutral]

**4. VALUATION SCENARIOS**

**DCF-Based Fair Value** (if available from analysts):
- Bear Case: $XX.XX (assumptions: [key assumptions])
- Base Case: $XX.XX (assumptions: [key assumptions])
- Bull Case: $XX.XX (assumptions: [key assumptions])

**Sum-of-Parts** (if conglomerate):
| Segment | Value | Method | Multiple |
|---------|-------|--------|----------|
| [Segment 1] | $XXB | EV/EBITDA | XXx |
| [Segment 2] | $XXB | EV/Revenue | XXx |
| Net Debt | -$XXB | | |
| **Fair Value** | $XXB | | $XX/share |

**5. KEY VALUATION DEBATES**
- Bulls argue: [Main valuation argument for higher multiples]
- Bears argue: [Main valuation argument for lower multiples]
- Key variable: [What metric would settle the debate]

Cite all sources, especially for analyst ratings, price targets, and valuation multiples."""

# --- INSTITUTIONAL OWNERSHIP PROMPT (NEW) ---
INSTITUTIONAL_OWNERSHIP_PROMPT = """Research institutional ownership and fund flows for {ticker_or_company}.

**1. OWNERSHIP STRUCTURE**

**Summary**:
- Institutional Ownership: XX.X%
- Retail Ownership: XX.X%
- Insider Ownership: XX.X%
- Float: XXX.XM shares ($XX.XB)

**Top Institutional Holders** (Latest 13F):
| Rank | Institution | Shares (M) | % of Float | Change QoQ | Value ($M) |
|------|-------------|------------|------------|------------|------------|
| 1 | [Fund Name] | XX.X | X.X% | +/-X.X% | $XXX |
| 2 | [Fund Name] | XX.X | X.X% | +/-X.X% | $XXX |
| 3 | [Fund Name] | XX.X | X.X% | +/-X.X% | $XXX |
| 4 | [Fund Name] | XX.X | X.X% | +/-X.X% | $XXX |
| 5 | [Fund Name] | XX.X | X.X% | +/-X.X% | $XXX |

**Ownership Concentration**:
- Top 10 holders own: XX.X% of float
- Top 25 holders own: XX.X% of float

**2. RECENT INSTITUTIONAL ACTIVITY** (Last Quarter)

**Largest Buys**:
| Institution | Shares Added | % Increase | New Position? |
|-------------|--------------|------------|---------------|
| [Fund Name] | +X.XM | +XX% | Yes/No |
| [Fund Name] | +X.XM | +XX% | Yes/No |

**Largest Sells**:
| Institution | Shares Sold | % Decrease | Exited? |
|-------------|-------------|------------|---------|
| [Fund Name] | -X.XM | -XX% | Yes/No |
| [Fund Name] | -X.XM | -XX% | Yes/No |

**Net Institutional Buying/Selling**: +/-X.XM shares ($XXM)

**3. NOTABLE HOLDER ANALYSIS**

**Active Managers with High Conviction**:
- [Fund]: XX.X% of their portfolio | [Investment style] | [Recent commentary if any]

**Hedge Fund Activity**:
- Notable hedge funds holding: [List]
- Recent hedge fund moves: [Any significant changes]

**Index Fund Ownership**:
- [List major index funds and weights]
- Index inclusion/exclusion risk: [Any upcoming changes]

**4. 13F FILING TIMELINE**
- Most recent 13F data: Q[X] [YYYY] (filed [date])
- Next 13F deadline: [date]
- Data lag caveat: Positions may have changed since filing

**5. ACTIVIST INVESTOR INVOLVEMENT**
- Current activists: [Name, stake, demands] or "None identified"
- Historical activism: [Any past campaigns]
- 13D filings: [Any recent 5%+ ownership disclosures]

**OWNERSHIP QUALITY ASSESSMENT**:
- Long-term holders %: XX%
- Turnover: [Low/Medium/High]
- Quality Score: [Assessment based on holder reputation]

Cite SEC filings (13F, 13D, 13G) and ownership data providers as sources."""

# --- SHORT INTEREST & SENTIMENT PROMPT (NEW) ---
SHORT_INTEREST_SENTIMENT_PROMPT = """Research short interest and market sentiment for {ticker_or_company}.

**1. SHORT INTEREST DATA**

**Current Metrics**:
| Metric | Current | Prior Month | 3 Months Ago | Assessment |
|--------|---------|-------------|--------------|------------|
| Short Interest | XX.XM shares | XX.XM | XX.XM | ↑/↓ |
| % of Float | XX.X% | XX.X% | XX.X% | ↑/↓ |
| Days to Cover | X.X days | X.X days | X.X days | ↑/↓ |
| % of Shares Outstanding | XX.X% | XX.X% | XX.X% | ↑/↓ |

**Short Interest Trend**:
- 6-month trend: [Increasing/Decreasing/Stable]
- Recent changes: [Any notable spikes or drops]

**Comparison to Peers**:
| Company | Short % of Float | Days to Cover |
|---------|-----------------|---------------|
| {ticker_or_company} | XX.X% | X.X |
| [Peer 1] | XX.X% | X.X |
| [Peer 2] | XX.X% | X.X |
| Sector Avg | XX.X% | X.X |

**Short Squeeze Risk**: [Low/Medium/High] — [Rationale]

**2. OPTIONS MARKET SENTIMENT**

**Put/Call Ratio**:
- Current: X.XX (vs 30-day avg: X.XX)
- Interpretation: [Bullish/Bearish/Neutral]

**Notable Options Activity** (Last 5 trading days):
- [Date]: [Unusual activity description, strike, expiry, volume]
- [Date]: [Unusual activity description, strike, expiry, volume]

**Implied Volatility**:
- Current IV: XX%
- IV Rank (52-week): XX% (XX = lowest, 100 = highest)
- Upcoming events priced in: [Earnings, etc.]

**3. SOCIAL MEDIA & RETAIL SENTIMENT**

**Reddit/WallStreetBets**:
- Mention frequency: [High/Medium/Low/None]
- Sentiment: [Bullish/Bearish/Mixed]
- Recent notable posts: [If any]

**StockTwits/Twitter**:
- Message volume trend: [Increasing/Decreasing]
- Sentiment score: [If available]

**Retail Trading Platforms**:
- Robinhood popularity rank: #XXX (if available)
- Recent retail flow: [Net buying/selling]

**4. ANALYST SENTIMENT INDICATORS**

**Earnings Estimate Revisions** (proxy for sentiment):
- Upward revisions (last 30 days): X analysts
- Downward revisions (last 30 days): X analysts
- Net revision momentum: [Positive/Negative]

**5. SENTIMENT SUMMARY**

| Indicator | Signal | Confidence |
|-----------|--------|------------|
| Short Interest | Bullish/Bearish/Neutral | High/Medium/Low |
| Options Flow | Bullish/Bearish/Neutral | High/Medium/Low |
| Social Sentiment | Bullish/Bearish/Neutral | High/Medium/Low |
| Analyst Revisions | Bullish/Bearish/Neutral | High/Medium/Low |
| **Overall** | Bullish/Bearish/Neutral | |

**Contrarian Signals**: [Any extreme readings that might indicate contrarian opportunity]

Cite FINRA short interest data, options data providers, and sentiment platforms."""

# --- ESG & SUSTAINABILITY PROMPT (NEW) ---
ESG_SUSTAINABILITY_PROMPT = """Research ESG (Environmental, Social, Governance) profile for {ticker_or_company}.

**1. ESG RATINGS OVERVIEW**

| Provider | Rating/Score | Percentile | Trend |
|----------|--------------|------------|-------|
| MSCI | [AAA-CCC] | Top XX% | ↑/↓/→ |
| Sustainalytics | [Score] / [Risk Level] | Top XX% | ↑/↓/→ |
| S&P Global | [Score] | Top XX% | ↑/↓/→ |
| ISS | [Score] | [Quartile] | ↑/↓/→ |

**Rating Trend**: [Improving/Declining/Stable]
**vs Peers**: [Above/Below/In-line with] sector average

**2. ENVIRONMENTAL**

**Climate & Carbon**:
- Scope 1+2 Emissions: XX,XXX tonnes CO2e
- Emissions Intensity: XX tonnes/$M revenue
- Net Zero Target: [Year] or "No commitment"
- Science-Based Target: [Yes/No]
- Carbon trajectory: [On track/Behind]

**Environmental Risks**:
- Physical climate risk exposure: [Assessment]
- Transition risk exposure: [Assessment]
- Stranded asset risk: [If applicable]

**Environmental Initiatives**:
- [Key initiative 1]
- [Key initiative 2]

**Environmental Controversies**: [Any recent issues]

**3. SOCIAL**

**Workforce**:
- Total Employees: XXX,XXX
- Turnover Rate: XX%
- Employee Satisfaction: [Glassdoor rating, surveys]
- DEI Metrics:
  - Women in workforce: XX%
  - Women in leadership: XX%
  - Diversity initiatives: [Summary]

**Human Capital Issues**:
- Layoffs: [Recent layoffs, if any]
- Labor relations: [Union status, strikes]
- Workplace safety: [Incident rates]

**Supply Chain**:
- Supplier audits: [Status]
- Human rights due diligence: [Assessment]

**Customer/Community**:
- Product safety issues: [Any recalls]
- Data privacy concerns: [Any breaches]
- Community relations: [Assessment]

**Social Controversies**: [Any recent issues]

**4. GOVERNANCE**

**Board Quality**:
- Independence: XX% independent
- Diversity: XX% diverse
- Tenure: XX years average
- Overboarding: [Any concerns]
- Separation of Chair/CEO: [Yes/No]

**Shareholder Rights**:
- Dual-class shares: [Yes/No]
- Poison pill: [Yes/No]
- Vote standard: [Majority/Plurality]
- Proxy access: [Yes/No]

**Executive Compensation**:
- CEO Pay Ratio: XXX:1
- Pay for Performance alignment: [Assessment]
- Say on Pay result: XX% approval

**Ethics & Compliance**:
- Accounting controversies: [Any issues]
- Regulatory violations: [Any fines]
- Political spending transparency: [Assessment]

**Governance Controversies**: [Any recent issues]

**5. MATERIAL ESG ISSUES** (SASB Framework)

For {ticker_or_company}'s industry, the most material ESG issues are:
1. [Material Issue 1]: [Company performance]
2. [Material Issue 2]: [Company performance]
3. [Material Issue 3]: [Company performance]

**6. ESG INVESTMENT IMPLICATIONS**

**Inclusion in ESG Indices/Funds**:
- Included in: [List ESG indices]
- Excluded from: [List any exclusions and reasons]

**ESG-Related Risks to Monitor**:
1. [Risk 1]: [Potential impact on stock]
2. [Risk 2]: [Potential impact on stock]

**ESG Improvement Opportunities**:
- [Area where improvement could drive rerating]

**OVERALL ESG ASSESSMENT**
- Strengths: [Key positives]
- Weaknesses: [Key concerns]
- Trajectory: [Improving/Stable/Declining]
- Investment Relevance: [How material to investment case]

Cite ESG rating providers, company sustainability reports, and news sources."""

# --- SUPPLY CHAIN & KEY RELATIONSHIPS PROMPT (NEW) ---
SUPPLY_CHAIN_RELATIONSHIPS_PROMPT = """Research supply chain and key business relationships for {ticker_or_company}.

**1. CUSTOMER CONCENTRATION**

**Top Customers** (if disclosed):
| Customer | % of Revenue | Relationship | Contract Status |
|----------|--------------|--------------|-----------------|
| [Customer 1] | XX% | [Years] | [Renewal date if known] |
| [Customer 2] | XX% | [Years] | [Renewal date if known] |
| [Customer 3] | XX% | [Years] | [Renewal date if known] |

**Concentration Risk**:
- Top customer: XX% of revenue
- Top 5 customers: XX% of revenue
- Top 10 customers: XX% of revenue

**Recent Customer Changes**:
- Won: [Notable new customers]
- Lost: [Notable churned customers]

**2. SUPPLIER CONCENTRATION**

**Critical Suppliers**:
| Supplier | Input | % of COGS | Alternatives | Risk |
|----------|-------|-----------|--------------|------|
| [Supplier 1] | [Component] | XX% | [Few/Many] | H/M/L |
| [Supplier 2] | [Component] | XX% | [Few/Many] | H/M/L |

**Single-Source Dependencies**:
- [List any components with single supplier]

**Geographic Supply Risk**:
- XX% of supply from [Country/Region]
- Exposure to [tariffs/geopolitical risks]

**Recent Supply Issues**:
- [Any recent disruptions or concerns]

**3. KEY PARTNERSHIPS**

**Strategic Partnerships**:
| Partner | Type | Start Date | Significance |
|---------|------|------------|--------------|
| [Partner 1] | [JV/License/Distribution] | [Year] | [Importance] |
| [Partner 2] | [JV/License/Distribution] | [Year] | [Importance] |

**Partnership Health**:
- [Assessment of key partnership stability]
- [Any partnership at risk]

**4. DISTRIBUTION CHANNELS**

**Revenue by Channel**:
| Channel | % of Revenue | Trend | Margin Profile |
|---------|--------------|-------|----------------|
| Direct | XX% | ↑/↓ | Higher/Lower |
| Retail Partners | XX% | ↑/↓ | Higher/Lower |
| Distributors | XX% | ↑/↓ | Higher/Lower |
| E-commerce | XX% | ↑/↓ | Higher/Lower |

**Key Retail/Distribution Partners**:
- [Partner]: XX% of sales, [relationship status]

**5. GEOGRAPHIC REVENUE MIX**

| Region | % of Revenue | Growth | Risk Factors |
|--------|--------------|--------|--------------|
| North America | XX% | XX% | [Key risks] |
| Europe | XX% | XX% | [Key risks] |
| Asia-Pacific | XX% | XX% | [Key risks] |
| Other | XX% | XX% | [Key risks] |

**Geographic Concentration Risk**: [Assessment]

**6. CONTRACT BACKLOG** (if applicable)

- Current Backlog: $X.XB
- Book-to-Bill Ratio: X.Xx
- Backlog Duration: X.X years
- YoY Change: +/-XX%

**7. RELATIONSHIP RISK SUMMARY**

| Relationship Type | Concentration Risk | Mitigation |
|-------------------|-------------------|------------|
| Customers | Low/Medium/High | [Actions] |
| Suppliers | Low/Medium/High | [Actions] |
| Partners | Low/Medium/High | [Actions] |
| Geographic | Low/Medium/High | [Actions] |

**Key Relationship Monitoring Events**:
- [Upcoming contract renewals]
- [Partner agreement expirations]
- [Regulatory changes affecting relationships]

Cite 10-K filings, earnings transcripts, and company disclosures for concentration data."""

# --- MACRO SENSITIVITY PROMPT (NEW) ---
MACRO_SENSITIVITY_PROMPT = """Research macroeconomic sensitivity and scenario analysis for {ticker_or_company}.

**1. ECONOMIC SENSITIVITY PROFILE**

**Revenue Sensitivity**:
| Macro Factor | Sensitivity | Evidence |
|--------------|-------------|----------|
| GDP Growth | High/Medium/Low | [Correlation or beta] |
| Consumer Spending | High/Medium/Low | [Evidence] |
| Business Investment | High/Medium/Low | [Evidence] |
| Housing Market | High/Medium/Low | [Evidence] |
| Interest Rates | High/Medium/Low | [Evidence] |

**Cyclicality Assessment**:
- Business Cycle Sensitivity: [Early/Mid/Late cycle, or Defensive]
- Historical performance in recessions: [Data from 2008, 2020]
- Revenue decline in last recession: XX%
- Recovery pattern: [V-shaped/U-shaped/L-shaped]

**2. INTEREST RATE SENSITIVITY**

**Direct Impact**:
- Debt Profile: $X.XB total, XX% floating rate
- Interest Expense Sensitivity: +100bps = $XXM additional expense
- Net Interest Income (if financial): [Sensitivity]

**Indirect Impact**:
- Customer financing sensitivity: [If applicable]
- Valuation multiple sensitivity: [Growth stocks more sensitive]
- Competition for capital: [Assessment]

**3. CURRENCY EXPOSURE**

**Revenue by Currency**:
| Currency | % of Revenue | Hedged % | Net Exposure |
|----------|--------------|----------|--------------|
| USD | XX% | - | - |
| EUR | XX% | XX% | XX% unhedged |
| [Other] | XX% | XX% | XX% unhedged |

**Currency Impact** (Last Year):
- FX headwind/tailwind: $XXM or XX% of revenue
- Translation vs Transaction exposure: [Split]

**Sensitivity**: 10% USD strengthening = XX% revenue headwind

**4. COMMODITY EXPOSURE**

**Key Commodity Inputs**:
| Commodity | Annual Usage | % of COGS | Hedged | Sensitivity |
|-----------|--------------|-----------|--------|-------------|
| [Commodity 1] | [Volume] | XX% | XX% | +10% = $XXM cost |
| [Commodity 2] | [Volume] | XX% | XX% | +10% = $XXM cost |

**Pricing Power**: [Can pass through commodity costs?]

**5. INFLATION SENSITIVITY**

**Cost Inflation**:
- Labor (% of costs): XX% — recent wage inflation: XX%
- Materials (% of costs): XX% — recent inflation: XX%
- Other operating costs: XX%

**Pricing Power**:
- Historical price increases: XX% annually
- Customer price sensitivity: [Assessment]
- Contract escalators: [If applicable]

**Net Margin Sensitivity to Inflation**: [Assessment]

**6. REGULATORY & POLICY SENSITIVITY**

**Tax Sensitivity**:
- Effective Tax Rate: XX%
- Geographic profit mix: [If low-tax jurisdictions at risk]
- R&D credit dependency: [If significant]
- Sensitivity: +1% tax rate = $XXM EPS impact

**Policy Risks**:
- [Specific policies that could impact: tariffs, regulation, etc.]
- Probability of adverse change: [Assessment]
- Potential impact: [Quantified]

**7. SCENARIO ANALYSIS**

**Recession Scenario**:
- Revenue impact: -XX% to -XX%
- Margin impact: -XXXbps to -XXXbps
- Historical precedent: [2008/2020 performance]

**Stagflation Scenario**:
- Revenue impact: [Assessment]
- Margin impact: [Assessment — cost inflation vs pricing power]

**Strong Growth Scenario**:
- Revenue upside: +XX%
- Operating leverage: +XXXbps margin expansion

**8. MACRO HEDGING**

**Natural Hedges**:
- [Geographic diversification]
- [Cost structure flexibility]
- [Pricing mechanisms]

**Financial Hedges**:
- FX hedges in place: [Coverage and tenor]
- Commodity hedges: [Coverage and tenor]
- Interest rate hedges: [Coverage and tenor]

**MACRO SENSITIVITY SUMMARY**

| Factor | Exposure | Current Direction | Impact |
|--------|----------|-------------------|--------|
| Economic Cycle | High/Medium/Low | Expansion/Contraction | +/-XX% |
| Interest Rates | High/Medium/Low | Rising/Falling | +/-XX% |
| Currency | High/Medium/Low | USD Strong/Weak | +/-XX% |
| Inflation | High/Medium/Low | High/Moderate | +/-XX% |
| Policy | High/Medium/Low | Favorable/Adverse | +/-XX% |

**Overall Macro Risk**: [High/Medium/Low] with current environment [favorable/unfavorable]

Cite 10-K risk factors, earnings commentary, and economic analysis for sensitivities."""

# --- PROMPT REGISTRY ---
# All available prompts with metadata
PROMPT_REGISTRY = {
    "RECENT_NEWS": {
        "template": RECENT_NEWS_PROMPT,
        "model": "sonar-pro",
        "description": "Recent news and developments",
        "typical_params": ["ticker_or_company", "days"],
    },
    "EARNINGS_ANALYSIS": {
        "template": EARNINGS_ANALYSIS_PROMPT,
        "model": "sonar-pro",
        "description": "Quarterly earnings analysis",
        "typical_params": ["ticker_or_company"],
    },
    "COMPETITIVE_LANDSCAPE": {
        "template": COMPETITIVE_LANDSCAPE_PROMPT,
        "model": "sonar-pro",
        "description": "Competitive positioning and market dynamics",
        "typical_params": ["ticker_or_company"],
    },
    "RISK_FACTORS": {
        "template": RISK_FACTORS_PROMPT,
        "model": "sonar-reasoning-pro",
        "description": "Comprehensive risk analysis",
        "typical_params": ["ticker_or_company"],
    },
    "BULL_BEAR_CASES": {
        "template": BULL_BEAR_CASES_PROMPT,
        "model": "sonar-reasoning-pro",
        "description": "Bull and bear investment cases",
        "typical_params": ["ticker_or_company"],
    },
    "MANAGEMENT_GOVERNANCE": {
        "template": MANAGEMENT_GOVERNANCE_PROMPT,
        "model": "sonar-pro",
        "description": "Management quality and corporate governance",
        "typical_params": ["ticker_or_company"],
    },
    "VALUATION_ANALYST": {
        "template": VALUATION_ANALYST_PROMPT,
        "model": "sonar-pro",
        "description": "Valuation metrics and analyst sentiment",
        "typical_params": ["ticker_or_company"],
    },
    "INSTITUTIONAL_OWNERSHIP": {
        "template": INSTITUTIONAL_OWNERSHIP_PROMPT,
        "model": "sonar-pro",
        "description": "Institutional ownership and fund flows",
        "typical_params": ["ticker_or_company"],
    },
    "SHORT_INTEREST_SENTIMENT": {
        "template": SHORT_INTEREST_SENTIMENT_PROMPT,
        "model": "sonar-pro",
        "description": "Short interest and market sentiment",
        "typical_params": ["ticker_or_company"],
    },
    "ESG_SUSTAINABILITY": {
        "template": ESG_SUSTAINABILITY_PROMPT,
        "model": "sonar-pro",
        "description": "ESG profile and sustainability",
        "typical_params": ["ticker_or_company"],
    },
    "SUPPLY_CHAIN_RELATIONSHIPS": {
        "template": SUPPLY_CHAIN_RELATIONSHIPS_PROMPT,
        "model": "sonar-pro",
        "description": "Supply chain and key business relationships",
        "typical_params": ["ticker_or_company"],
    },
    "MACRO_SENSITIVITY": {
        "template": MACRO_SENSITIVITY_PROMPT,
        "model": "sonar-reasoning-pro",
        "description": "Macroeconomic sensitivity analysis",
        "typical_params": ["ticker_or_company"],
    },
}