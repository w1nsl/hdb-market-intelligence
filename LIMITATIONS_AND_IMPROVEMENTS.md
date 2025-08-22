# Limitations and Future Improvements

## Current Limitations

### 1. Data Features
- **Limited property attributes**: Currently only uses basic features (town, flat type, floor area, storey, model, lease info)
- **No location intelligence**: Missing proximity data to amenities, schools, transport hubs
- **Static economic context**: No macroeconomic indicators affecting housing market
- **No real-time market sentiment**: Missing current market trends and buyer behavior

### 2. LLM Workflow
- **Simple conversation flow**: Linear workflow with basic routing logic
- **Limited context awareness**: Cannot handle complex multi-turn conversations about related topics
- **No memory persistence**: Conversations don't retain long-term context across sessions
- **Basic intent classification**: Simple keyword-based routing instead of sophisticated NLU

### 3. User Experience
- **Basic web interface**: Simple chat UI without rich interactions or visualizations
- **No data visualization**: Results presented as text only, no charts or maps

## Proposed Improvements

#### Location Intelligence
- **OneMap API Integration** (`https://www.onemap.gov.sg/apidocs/maps`)
  - Distance to nearest MRT stations, bus stops
  - Proximity to schools (primary, secondary, JC, universities)
  - Healthcare facilities (hospitals, polyclinics)
  - Shopping malls, markets, recreational facilities
  - Park connectors and green spaces

- **LTA DataMall Integration**
  - Real-time transport accessibility scores
  - Bus route density and frequency
  - Traffic patterns and congestion data
  - Future transport infrastructure plans

- **URA Master Plan Data**
  - Zoning information and future development plans
  - Plot ratio and building height restrictions
  - Upcoming infrastructure projects
  - Land use changes and urban planning insights

#### Macroeconomic Indicators
- **MAS Statistics Integration**
  - Interest rates (SIBOR, bank lending rates)
  - Money supply and liquidity metrics
  - Property market cooling measures timeline

- **SingStat Data**
  - Consumer Price Index (CPI)
  - GDP growth rates
  - Employment statistics
  - Population demographics and migration patterns
  - Household income distribution

#### Enhanced Intent Understanding
- **Multi-intent recognition**: Handle queries with multiple objectives
- **Contextual disambiguation**: Resolve ambiguous references using conversation history

#### Data Visualisation
- **Comparative analysis**: Side-by-side property and area comparisons
- **Neighborhood insights**: Detailed area profiles with amenities mapping
