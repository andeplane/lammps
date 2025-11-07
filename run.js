#!/usr/bin/env node

/**
 * Simple Node.js test script for LAMMPS compiled with Emscripten
 * 
 * Usage: node run.js [input_file]
 * Default input file: lj.in
 */

const fs = require('fs');
const path = require('path');

// Load the LAMMPS module factory function
const createModule = require('./build_kokkos/lmp.js');

console.log('Loading LAMMPS module...');

// Create and initialize the module - returns a Promise with pthreads
createModule({
  // Module configuration options if needed
  print: (text) => process.stdout.write(text + '\n'),
  printErr: (text) => process.stderr.write(text + '\n'),
}).then((Module) => {
  console.log('LAMMPS module loaded successfully');
  
  // Get input file (default to lj.in)
  const inputFile = process.argv[2] || 'lj.in';
  
  if (!fs.existsSync(inputFile)) {
    console.error(`Error: Input file '${inputFile}' not found`);
    process.exit(1);
  }
  
  console.log(`Running LAMMPS with input file: ${inputFile}`);
  
  // Read input file content
  const inputContent = fs.readFileSync(inputFile, 'utf8');
  
  // Initialize LAMMPS with Kokkos enabled (-k on)
  // We need to create an argv array: ["lmp", "-k", "on"]
  const args = ["lmp", "-k", "on"];
  const argc = args.length;
  
  // Allocate each string
  const stringPtrs = [];
  for (let i = 0; i < argc; i++) {
    const strPtr = Module.allocateUTF8(args[i]);
    stringPtrs.push(BigInt(strPtr));
  }
  
  // Create a buffer for argv (array of pointers) - allocate as dummy string space
  const ptrArraySize = argc * 8;  // 8 bytes per pointer with MEMORY64
  const dummyString = '\0'.repeat(ptrArraySize);
  const argvPtr = Module.allocateUTF8(dummyString);
  
  // Write the string pointers into this buffer
  for (let i = 0; i < argc; i++) {
    Module.HEAP64[argvPtr / 8 + i] = stringPtrs[i];
  }
  
  const lmp = Module._lammps_open_no_mpi(argc, BigInt(argvPtr), 0n);
  
  if (!lmp) {
    console.error('Error: Failed to create LAMMPS instance');
    process.exit(1);
  }
  
  console.log('LAMMPS instance created with Kokkos enabled (-k on)');
  
  // Execute commands directly from the input file content
  // lammps_commands_string processes multiple commands separated by newlines
  const commandsPtr = BigInt(Module.allocateUTF8(inputContent));
  Module._lammps_commands_string(lmp, commandsPtr);
  // Note: We don't free manually - Emscripten handles cleanup
  
  console.log('LAMMPS simulation completed');
  
  // Clean up
  Module._lammps_close(lmp);
  
  console.log('Done!');
}).catch((err) => {
  console.error('Error loading module:', err);
  process.exit(1);
});

