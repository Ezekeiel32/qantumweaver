const express = require('express');
const archiver = require('archiver');
const fs = require('fs');
const path = require('path');
const { mkdirSync, rmSync } = require('fs');
const stream = require('stream');
const { promisify } = require('util');
const pipeline = promisify(stream.pipeline);
const cors = require('cors');
const { spawn } = require('child_process');

const app = express();
const PORT = 9001; // Or any port you want

app.use(cors());

process.on('uncaughtException', function (err) {
  console.error('Uncaught Exception:', err);
});
process.on('unhandledRejection', function (reason, promise) {
  console.error('Unhandled Rejection:', reason);
});

async function fetchAllFilesRecursive(datasetId, branch, dir = '') {
  const apiUrl = `https://huggingface.co/api/datasets/${datasetId}/tree/${branch}/${dir}`;
  const res = await fetch(apiUrl);
  if (!res.ok) return [];
  const files = await res.json();
  let allFiles = [];
  for (const file of files) {
    if (file.type === 'file') {
      allFiles.push({ ...file, path: dir ? `${dir}/${file.path}` : file.path });
    } else if (file.type === 'directory') {
      const subFiles = await fetchAllFilesRecursive(datasetId, branch, dir ? `${dir}/${file.path}` : file.path);
      allFiles = allFiles.concat(subFiles);
    }
  }
  return allFiles;
}

async function downloadAllHuggingFaceFiles(datasetId, tempDir) {
  let files = await fetchAllFilesRecursive(datasetId, 'main');
  if (!files || files.length === 0) files = await fetchAllFilesRecursive(datasetId, 'master');
  if (!files || files.length === 0) throw new Error('No files found in HuggingFace dataset (tried both main and master branches).');
  for (const file of files) {
    let fileUrl = `https://huggingface.co/datasets/${datasetId}/resolve/main/${file.path}`;
    let fileRes = await fetch(fileUrl);
    if (!fileRes.ok) {
      fileUrl = `https://huggingface.co/datasets/${datasetId}/resolve/master/${file.path}`;
      fileRes = await fetch(fileUrl);
    }
    if (!fileRes.ok) {
      console.error('Failed to download file:', file.path);
      continue;
    }
    const filePath = path.join(tempDir, file.path);
    mkdirSync(path.dirname(filePath), { recursive: true });
    try {
      await pipeline(fileRes.body, fs.createWriteStream(filePath));
    } catch (err) {
      console.error('Failed to write file:', filePath, err);
    }
  }
}

app.get('/download-hf', async (req, res) => {
  const { dataset } = req.query;
  if (!dataset) {
    console.error('Missing dataset parameter');
    return res.status(400).json({ error: 'Missing dataset parameter' });
  }
  const tempDir = path.join('/tmp', `hf_${dataset.replace(/[\\/:*?"<>|]/g, '_')}_${Date.now()}`);
  mkdirSync(tempDir, { recursive: true });
  try {
    await downloadAllHuggingFaceFiles(dataset, tempDir);
    res.setHeader('Content-Type', 'application/zip');
    res.setHeader('Content-Disposition', `attachment; filename="${dataset.replace(/[\\/:*?"<>|]/g, '_')}.zip"`);
    const archive = archiver('zip', { zlib: { level: 9 } });
    archive.directory(tempDir, false);
    archive.finalize();
    archive.pipe(res);
    archive.on('end', () => rmSync(tempDir, { recursive: true, force: true }));
    archive.on('error', err => {
      console.error('Archiver error:', err);
      rmSync(tempDir, { recursive: true, force: true });
      res.status(500).json({ error: err.message });
    });
  } catch (err) {
    console.error('Route error:', err);
    rmSync(tempDir, { recursive: true, force: true });
    res.status(500).json({ error: err.message });
  }
});

app.get('/download-github', async (req, res) => {
  const { repo } = req.query;
  if (!repo) {
    console.error('Missing repo parameter');
    return res.status(400).json({ error: 'Missing repo parameter' });
  }
  const branches = ['main', 'master'];
  let lastError = null;
  for (const branch of branches) {
    const zipUrl = `https://github.com/${repo}/archive/refs/heads/${branch}.zip`;
    try {
      const response = await fetch(zipUrl);
      if (response.ok) {
        res.setHeader('Content-Type', 'application/zip');
        res.setHeader('Content-Disposition', `attachment; filename="${repo.replace('/', '_')}_${branch}.zip"`);
        await pipeline(response.body, res);
        return;
      } else {
        lastError = `Failed to fetch GitHub ZIP for branch ${branch}: ${response.statusText}`;
        console.error(lastError);
      }
    } catch (err) {
      lastError = `GitHub download error for branch ${branch}: ${err.message}`;
      console.error(lastError);
    }
  }
  res.status(500).json({ error: lastError || 'Failed to download GitHub repo ZIP from both main and master branches.' });
});

app.get('/download-kaggle', async (req, res) => {
  const { ref } = req.query;
  if (!ref) {
    return res.status(400).json({ error: 'Missing dataset ref parameter' });
  }

  const tempDir = path.join('/tmp', `kaggle_${ref.replace(/[^a-z0-9_.-]/gi, '_')}_${Date.now()}`);
  mkdirSync(tempDir, { recursive: true });

  try {
    const pythonExecutable = path.join(process.cwd(), '.venv', 'bin', 'python');
    const scriptPath = path.join(process.cwd(), 'scripts', 'download_kaggle.py');
    const scriptArgs = ['--ref', ref, '--dest', tempDir];

    const pyProc = spawn(pythonExecutable, [scriptPath, ...scriptArgs]);

    pyProc.on('close', (code) => {
      if (code === 0) {
        res.setHeader('Content-Type', 'application/zip');
        res.setHeader('Content-Disposition', `attachment; filename="${ref.replace('/', '_')}.zip"`);
        const archive = archiver('zip', { zlib: { level: 9 } });
        
        archive.on('end', () => rmSync(tempDir, { recursive: true, force: true }));
        archive.on('error', err => {
          console.error('Archiver error:', err);
          rmSync(tempDir, { recursive: true, force: true });
          if (!res.headersSent) {
            res.status(500).json({ error: err.message });
          }
        });
        
        archive.pipe(res);
        archive.directory(tempDir, false);
        archive.finalize();
      } else {
        rmSync(tempDir, { recursive: true, force: true });
        res.status(500).json({ error: `Kaggle download script failed with code ${code}` });
      }
    });

    pyProc.stderr.on('data', (data) => {
      console.error(`Kaggle Script STDERR: ${data}`);
    });

  } catch (err) {
    rmSync(tempDir, { recursive: true, force: true });
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`Dataset download server running on http://localhost:${PORT}`);
}); 