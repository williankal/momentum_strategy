const { getHistoricalRates } = require("dukascopy-node");
const fs = require('fs');
const path = require('path');

// Set date range from 2021 to present
const startDate = new Date("2021-01-01");
const endDate = new Date(); // Current date

// Format dates for display
const formatDate = (date) => {
  return date.toISOString().split('T')[0];
};

// List of cryptocurrencies with USD pairs only
const cryptos = [
  { name: "Cardano vs US Dollar", symbol: "adausd" },
  { name: "Aave vs US Dollar", symbol: "aveusd" },
  { name: "Basic Attention Token vs US Dollar", symbol: "batusd" },
  { name: "Bitcoin Cash vs US dollar", symbol: "bchusd" },
  { name: "Bitcoin vs US Dollar", symbol: "btcusd" },
  { name: "Compound vs US Dollar", symbol: "cmpusd" },
  { name: "Dashcoin vs US Dollar", symbol: "dshusd" },
  { name: "Enjin vs US Dollar", symbol: "enjusd" },
  { name: "EOS vs US Dollar", symbol: "eosusd" },
  { name: "Ether vs US Dollar", symbol: "ethusd" },
  { name: "Chainlink vs US Dollar", symbol: "lnkusd" },
  { name: "Litecoin vs US Dollar", symbol: "ltcusd" },
  { name: "Polygon vs US Dollar", symbol: "matusd" },
  { name: "Maker vs US Dollar", symbol: "mkrusd" },
  { name: "TRON vs US Dollar", symbol: "trxusd" },
  { name: "Uniswap vs US Dollar", symbol: "uniusd" },
  { name: "Stellar vs US Dollar", symbol: "xlmusd" },
  { name: "Yearn.finance vs US Dollar", symbol: "yfiusd" }
];

// Batch settings to avoid rate limiting
const BATCH_SIZE = 5;
const PAUSE_BETWEEN_CRYPTOS_MS = 5000; // 5 seconds pause between crypto downloads

// Ensure the data/hourly directory exists
const outputDir = 'data/hourly';
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true });
  console.log(`Created directory: ${outputDir}`);
}

async function downloadAllCryptoData() {
  console.log(`Starting download for ${cryptos.length} cryptocurrency/USD pairs`);
  console.log(`Date range: ${formatDate(startDate)} to ${formatDate(endDate)}`);
  console.log(`Using batch size of ${BATCH_SIZE} with 1000ms pause between batches`);
  console.log(`Downloading hourly (h1) data instead of daily`);
  console.log(`Saving files to: ${outputDir}`);
  console.log("-----------------------------------------------------");

  const results = {};
  
  for (let i = 0; i < cryptos.length; i++) {
    const crypto = cryptos[i];
    console.log(`[${i+1}/${cryptos.length}] Downloading ${crypto.name} (${crypto.symbol})...`);
    
    try {
      const data = await getHistoricalRates({
        instrument: crypto.symbol,
        dates: {
          from: startDate,
          to: endDate,
        },
        timeframe: "h1", // Changed from d1 to h1 for hourly data
        format: "json", 
        batchSize: BATCH_SIZE,
        pauseBetweenBatchesMs: 1000,
      });
      
      results[crypto.symbol] = data;
      
      // Save to file in the data/hourly directory
      const filename = path.join(outputDir, `${crypto.symbol}_${formatDate(startDate)}_to_${formatDate(endDate)}.json`);
      fs.writeFileSync(filename, JSON.stringify(data, null, 2));
      console.log(`✓ Saved to ${filename}`);
      
      // Pause between crypto downloads to be nice to the server
      if (i < cryptos.length - 1) {
        console.log(`Pausing for ${PAUSE_BETWEEN_CRYPTOS_MS/1000} seconds before next download...`);
        await new Promise(resolve => setTimeout(resolve, PAUSE_BETWEEN_CRYPTOS_MS));
      }
    } catch (error) {
      console.error(`Error downloading ${crypto.symbol}:`, error.message);
    }
  }
  
  console.log("-----------------------------------------------------");
  console.log("Download process completed");
  return results;
}

// Execute the function
downloadAllCryptoData()
  .then(() => console.log("All data downloaded successfully!"))
  .catch(error => console.error("Main error:", error));