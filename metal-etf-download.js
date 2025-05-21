const { getHistoricalRates } = require("dukascopy-node");
const fs = require('fs');
const path = require('path');

// Data de início e fim (atual)
const startDate = new Date("2021-01-01");
const endDate = new Date(); // Data atual

// Função para formatar data (yyyy-mm-dd)
const formatDate = (date) => date.toISOString().split('T')[0];

// Lista de ativos com seus símbolos
const assets = [
  { name: "NY Cocoa", symbol: "cocoacmdusd" },
  { name: "Coffee Arabica", symbol: "coffeecmdusx" },
  { name: "Cotton", symbol: "cottoncmdusx" },
  { name: "Orange Juice", symbol: "ojuicecmdusx" },
  { name: "Soybean", symbol: "soybeancmdusx" },
  { name: "High Grade Copper", symbol: "coppercmdusd" },
  { name: "Palladium", symbol: "xpdcmdusd" },
  { name: "Spot silver", symbol: "xagusd" },
  { name: "Spot gold", symbol: "xauusd" },
  { name: "ARK 21Shares Active Bitcoin Ethereum Strategy ETF Fund", symbol: "arkiususd" },
  { name: "Global X Cybersecurity UCITS ETF Fund", symbol: "bugggbgbx" },
  { name: "Lyxor Smart Overnight Return - UCITS ETF C-GBP", symbol: "csh2gbgbx" },
  { name: "WisdomTree Cybersecurity UCITS ETF Fund", symbol: "cysegbgbx" },
  { name: "iShares MSCI Europe Health Care Sector UCITS ETF Fund", symbol: "esihgbgbx" },
  { name: "iShares Physical Gold ETC Fund", symbol: "iglnususd" },
  { name: "iShares S&P 500 Financials Sector UCITS ETF", symbol: "iufsususd" },
  { name: "iShares MSCI Global Semiconductors UCITS ETF", symbol: "semigbgbx" },
  { name: "Invesco Physical Gold ETC Fund", symbol: "sgldususd" },
  { name: "VanEck Semiconductor ETF Fund", symbol: "smhususd" },
  { name: "Lyxor Smart Overnight Return - UCITS ETF C-USD", symbol: "smtcususd" },
  { name: "Wisdomtree Artificial Intelligence And Innovation Fund", symbol: "wtaiususd" },
  { name: "Xtrackers FTSE Developed Europe Real Estate UCITS ETF", symbol: "xdergbgbx" },
  { name: "Xtrackers MSCI World Health Care UCITS ETF Fund", symbol: "xdwhususd" },
  { name: "Xtrackers MSCI World Information Technology UCITS ETF", symbol: "xdwtususd" },
  { name: "Invesco Real Estate S&P US Select Sector UCITS ETF Acc", symbol: "xresususd" },
];

// Configurações de batch e pausa
const BATCH_SIZE = 5;
const PAUSE_BETWEEN_ASSETS_MS = 5000;

// Diretório de saída
const outputDir = 'data/hourly';
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true });
  console.log(`Created directory: ${outputDir}`);
}

async function downloadAllAssetData() {
  console.log(`Starting download for ${assets.length} assets`);
  console.log(`Date range: ${formatDate(startDate)} to ${formatDate(endDate)}`);
  console.log(`Timeframe: hourly (h1)`);
  console.log(`Saving files to: ${outputDir}`);
  console.log("-----------------------------------------------------");

  const results = {};

  for (let i = 0; i < assets.length; i++) {
    const asset = assets[i];
    console.log(`[${i + 1}/${assets.length}] Downloading ${asset.name} (${asset.symbol})...`);

    try {
      const data = await getHistoricalRates({
        instrument: asset.symbol,
        dates: { from: startDate, to: endDate },
        timeframe: "h1",
        format: "json",
        batchSize: BATCH_SIZE,
        pauseBetweenBatchesMs: 1000,
      });

      results[asset.symbol] = data;

      const filename = path.join(outputDir, `${asset.symbol}_${formatDate(startDate)}_to_${formatDate(endDate)}.json`);
      fs.writeFileSync(filename, JSON.stringify(data, null, 2));
      console.log(`✓ Saved to ${filename}`);

      if (i < assets.length - 1) {
        console.log(`Pausing for ${PAUSE_BETWEEN_ASSETS_MS / 1000} seconds...`);
        await new Promise(resolve => setTimeout(resolve, PAUSE_BETWEEN_ASSETS_MS));
      }

    } catch (error) {
      console.error(`Error downloading ${asset.symbol}:`, error.message);
    }
  }

  console.log("-----------------------------------------------------");
  console.log("Download process completed.");
  return results;
}

// Executar a função
downloadAllAssetData()
  .then(() => console.log("All data downloaded successfully!"))
  .catch(error => console.error("Main error:", error));
