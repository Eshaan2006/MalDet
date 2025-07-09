import pefile
import numpy as np
import pandas as pd
import hashlib
import math
import re
import logging
from collections import Counter
import struct

class PEFeatureExtractor:
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        self.suspicious_sections = [
            '.textbss', '.packed', '.upx', '.aspack', '.rlpack',
            '.themida', '.vmp', '.vmprotect', '.enigma'
        ]
        
        self.suspicious_dlls = [
            'ws2_32.dll', 'wininet.dll', 'urlmon.dll', 'shell32.dll',
            'advapi32.dll', 'ntdll.dll', 'psapi.dll'
        ]

        self.suspicious_imports = [
            'CreateRemoteThread', 'VirtualAlloc', 'VirtualProtect', 'WriteProcessMemory'
        ]
        
        self.crypto_apis = [
            'CryptAcquireContext', 'CryptCreateHash', 'CryptHashData',
            'CryptDeriveKey', 'CryptEncrypt', 'CryptDecrypt'
        ]
        
        self.network_apis = [
            'InternetOpen', 'InternetConnect', 'HttpOpenRequest',
            'socket', 'connect', 'send', 'recv'
        ]
        
        self.process_apis = [
            'CreateProcess', 'OpenProcess', 'TerminateProcess',
            'VirtualAlloc', 'VirtualProtect', 'WriteProcessMemory'
        ]

    def extract_features(self, file_path):
        """Extract all features from a PE file"""
        try:
            pe = pefile.PE(file_path)
        except pefile.PEFormatError as e:
            self.logger.error(f"Invalid PE format: {file_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading PE file {file_path}: {e}")
            return None

        features = {}
        try:
            # Header, section, import, string, and metadata features
            features.update(self._extract_header_features(pe))
            features.update(self._extract_section_features(pe))
            features.update(self._extract_import_features(pe))
            features.update(self._extract_string_features(file_path))
            features.update(self._extract_metadata_features(pe, file_path))
        except Exception as e:
            self.logger.error(f"Error processing features for {file_path}: {e}")
            return None

        return features

    def _extract_header_features(self, pe):
        """Extract PE header-based features with safe fallbacks"""
        features = {}

        # DOS Header
        dos_header_size = pe.DOS_HEADER.sizeof()
        # Use getattr to safely access AddressOfNewExeHeader
        new_exe_header = getattr(pe.OPTIONAL_HEADER, 'AddressOfNewExeHeader', dos_header_size)
        features['dos_header_size'] = dos_header_size
        features['dos_stub_size'] = new_exe_header - dos_header_size

        # PE Header
        features['machine_type'] = pe.FILE_HEADER.Machine
        features['number_of_sections'] = pe.FILE_HEADER.NumberOfSections
        features['timestamp'] = pe.FILE_HEADER.TimeDateStamp
        features['size_of_optional_header'] = pe.FILE_HEADER.SizeOfOptionalHeader
        features['characteristics'] = pe.FILE_HEADER.Characteristics

        # Optional Header
        features['address_of_entry_point'] = getattr(pe.OPTIONAL_HEADER, 'AddressOfEntryPoint', 0)
        features['base_of_code'] = getattr(pe.OPTIONAL_HEADER, 'BaseOfCode', 0)
        features['image_base'] = getattr(pe.OPTIONAL_HEADER, 'ImageBase', 0)
        features['section_alignment'] = getattr(pe.OPTIONAL_HEADER, 'SectionAlignment', 0)
        features['file_alignment'] = getattr(pe.OPTIONAL_HEADER, 'FileAlignment', 0)
        features['size_of_image'] = getattr(pe.OPTIONAL_HEADER, 'SizeOfImage', 0)
        features['size_of_headers'] = getattr(pe.OPTIONAL_HEADER, 'SizeOfHeaders', 0)
        features['subsystem'] = getattr(pe.OPTIONAL_HEADER, 'Subsystem', 0)
        features['dll_characteristics'] = getattr(pe.OPTIONAL_HEADER, 'DllCharacteristics', 0)

        return features

    def _extract_section_features(self, pe):
        """Extract section-based features"""
        features = {}
        
        if not hasattr(pe, 'sections') or not pe.sections:
            return self._get_default_section_features()
        
        # Section counts and statistics
        executable_sections = 0
        writable_sections = 0
        entropies = []
        sizes = []
        virtual_ratios = []
        suspicious_names = 0
        high_entropy_count = 0
        rwx_count = 0
        zero_rawsize_count = 0
        
        for section in pe.sections:
            # Section characteristics
            is_executable = section.Characteristics & 0x20000000
            is_writable = section.Characteristics & 0x80000000
            is_readable = section.Characteristics & 0x40000000
            
            if is_executable:
                executable_sections += 1
            if is_writable:
                writable_sections += 1
            if is_executable and is_writable and is_readable:
                rwx_count += 1
                
            # Section entropy
            data = section.get_data()
            if len(data) > 0:
                entropy = self._calculate_entropy(data)
                entropies.append(entropy)
                if entropy > 7.5:
                    high_entropy_count += 1
            else:
                zero_rawsize_count += 1
                
            # Section sizes
            sizes.append(section.SizeOfRawData)
            if section.SizeOfRawData > 0:
                virtual_ratios.append(section.Misc_VirtualSize / section.SizeOfRawData)
            
            # Suspicious section names
            section_name = section.Name.decode('utf-8', errors='ignore').lower()
            if any(sus in section_name for sus in self.suspicious_sections):
                suspicious_names += 1
        
        # Aggregate statistics
        features['sections_executable_count'] = executable_sections
        features['sections_writable_count'] = writable_sections
        features['sections_entropy_mean'] = np.mean(entropies) if entropies else 0
        features['sections_entropy_std'] = np.std(entropies) if entropies else 0
        features['sections_size_mean'] = np.mean(sizes) if sizes else 0
        features['sections_size_std'] = np.std(sizes) if sizes else 0
        features['sections_virtual_size_ratio'] = np.mean(virtual_ratios) if virtual_ratios else 0
        features['sections_suspicious_names'] = suspicious_names
        features['sections_high_entropy_count'] = high_entropy_count
        features['sections_rwx_count'] = rwx_count
        features['sections_zero_rawsize'] = zero_rawsize_count
        
        # Section type ratios
        total_sections = len(pe.sections)
        features['code_section_ratio'] = executable_sections / total_sections if total_sections > 0 else 0
        features['data_section_ratio'] = writable_sections / total_sections if total_sections > 0 else 0
        features['resource_section_present'] = int(any('.rsrc' in str(s.Name) for s in pe.sections))
        
        return features

    def _extract_import_features(self, pe):
        """Extract import/export table features"""
        features = {}
        
        # Initialize counters
        dll_count = 0
        function_count = 0
        suspicious_dll_count = 0
        api_categories = {
            'crypto': 0, 'network': 0, 'process': 0,
            'registry': 0, 'file': 0, 'debug': 0
        }
        dll_ratios = {'kernel32': 0, 'ntdll': 0, 'advapi32': 0}
        
        # Process imports
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                dll_name = entry.dll.decode('utf-8', errors='ignore').lower()
                dll_count += 1
                
                if any(sus_dll in dll_name for sus_dll in self.suspicious_dlls):
                    suspicious_dll_count += 1
                
                # Count functions per DLL
                dll_functions = 0
                for imp in entry.imports:
                    function_count += 1
                    dll_functions += 1
                    
                    if imp.name:
                        func_name = imp.name.decode('utf-8', errors='ignore')
                        
                        # Categorize API calls
                        if any(api in func_name for api in self.crypto_apis):
                            api_categories['crypto'] += 1
                        if any(api in func_name for api in self.network_apis):
                            api_categories['network'] += 1
                        if any(api in func_name for api in self.process_apis):
                            api_categories['process'] += 1
                
                # Calculate ratios for important DLLs
                for key_dll in dll_ratios.keys():
                    if key_dll in dll_name:
                        dll_ratios[key_dll] = dll_functions
        
        # Normalize ratios
        if function_count > 0:
            for dll in dll_ratios:
                dll_ratios[dll] = dll_ratios[dll] / function_count
        
        # Store import features
        features['imports_dll_count'] = dll_count
        features['imports_function_count'] = function_count
        features['imports_suspicious_dll_count'] = suspicious_dll_count
        features['imports_api_ratio_kernel32'] = dll_ratios['kernel32']
        features['imports_api_ratio_ntdll'] = dll_ratios['ntdll']
        features['imports_api_ratio_advapi32'] = dll_ratios['advapi32']
        
        # API category flags
        features['imports_has_crypto_apis'] = int(api_categories['crypto'] > 0)
        features['imports_has_network_apis'] = int(api_categories['network'] > 0)
        features['imports_has_process_apis'] = int(api_categories['process'] > 0)
        
        # Export features
        export_count = 0
        if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT'):
            export_count = len(pe.DIRECTORY_ENTRY_EXPORT.symbols)
        
        features['exports_function_count'] = export_count
        
        return features

    def _extract_string_features(self, file_path):
        """Extract string-based features"""
        features = {}
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Extract strings (basic regex approach)
            ascii_strings = re.findall(b'[\x20-\x7E]{4,}', data)
            unicode_strings = re.findall(b'(?:[\x20-\x7E]\x00){4,}', data)
            
            # String statistics
            all_strings = ascii_strings + unicode_strings
            features['strings_count_total'] = len(all_strings)
            features['strings_count_printable'] = len(ascii_strings)
            features['strings_count_unicode'] = len(unicode_strings)
            
            if all_strings:
                string_lengths = [len(s) for s in all_strings]
                features['strings_length_mean'] = np.mean(string_lengths)
                features['strings_length_max'] = max(string_lengths)
            else:
                features['strings_length_mean'] = 0
                features['strings_length_max'] = 0
            
            # Pattern matching
            combined_text = b' '.join(all_strings).decode('utf-8', errors='ignore')
            features['strings_urls_count'] = len(re.findall(r'https?://[^\s]+', combined_text))
            features['strings_ips_count'] = len(re.findall(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', combined_text))
            features['strings_registry_keys'] = len(re.findall(r'HKEY_[A-Z_]+', combined_text))
            
            # High entropy strings
            high_entropy_count = 0
            for string in all_strings:
                if len(string) > 8:
                    entropy = self._calculate_entropy(string)
                    if entropy > 6.0:
                        high_entropy_count += 1
            
            features['strings_high_entropy_count'] = high_entropy_count
            
        except Exception as e:
            print(f"Error extracting strings: {e}")
            features = self._get_default_string_features()
        
        return features

    def _extract_metadata_features(self, pe, file_path):
        """Extract file metadata features"""
        features = {}
        
        try:
            import os
            
            # File size
            file_size = os.path.getsize(file_path)
            features['file_size'] = file_size
            
            # File entropy
            with open(file_path, 'rb') as f:
                data = f.read()
            features['file_entropy'] = self._calculate_entropy(data)
            
            # Size ratios
            virtual_size = pe.OPTIONAL_HEADER.SizeOfImage
            features['file_size_virtual_ratio'] = file_size / virtual_size if virtual_size > 0 else 0
            
            # Timestamp analysis
            import time
            current_time = time.time()
            compile_time = pe.FILE_HEADER.TimeDateStamp
            
            features['timestamp_anomaly'] = int(compile_time > current_time or compile_time < 946684800)  # Before 2000
            features['timestamp_future'] = int(compile_time > current_time)
            features['timestamp_too_old'] = int(compile_time < 946684800)
            
            # Overlay detection
            overlay_offset = pe.get_overlay_data_start_offset()
            if overlay_offset:
                overlay_size = file_size - overlay_offset
                features['overlay_size'] = overlay_size
                
                # Overlay entropy
                with open(file_path, 'rb') as f:
                    f.seek(overlay_offset)
                    overlay_data = f.read(min(8192, overlay_size))  # Sample first 8KB
                features['overlay_entropy'] = self._calculate_entropy(overlay_data)
            else:
                features['overlay_size'] = 0
                features['overlay_entropy'] = 0
            
            # Packing ratio (heuristic)
            features['packing_ratio'] = file_size / virtual_size if virtual_size > 0 else 1.0
            
        except Exception as e:
            print(f"Error extracting metadata: {e}")
            features = self._get_default_metadata_features()
        
        return features

    def _calculate_entropy(self, data):
        """Calculate Shannon entropy of data"""
        if not data:
            return 0
        
        # Count byte frequencies
        byte_counts = Counter(data)
        data_len = len(data)
        
        # Calculate entropy
        entropy = 0
        for count in byte_counts.values():
            if count > 0:
                probability = count / data_len
                entropy -= probability * math.log2(probability)
        
        return entropy

    def _get_default_features(self):
        """Return default feature values for failed extractions"""
        defaults = {}
        
        # Add all expected features with default values
        feature_names = [
            'dos_header_size', 'dos_stub_size', 'machine_type', 'number_of_sections',
            'timestamp', 'size_of_optional_header', 'characteristics',
            'address_of_entry_point', 'base_of_code', 'image_base', 'section_alignment',
            'file_alignment', 'size_of_image', 'size_of_headers', 'subsystem',
            'dll_characteristics', 'sections_executable_count', 'sections_writable_count',
            'sections_entropy_mean', 'sections_entropy_std', 'sections_size_mean',
            'sections_size_std', 'sections_virtual_size_ratio', 'sections_suspicious_names',
            'sections_high_entropy_count', 'sections_rwx_count', 'sections_zero_rawsize',
            'code_section_ratio', 'data_section_ratio', 'resource_section_present',
            'imports_dll_count', 'imports_function_count', 'imports_suspicious_dll_count',
            'imports_api_ratio_kernel32', 'imports_api_ratio_ntdll', 'imports_api_ratio_advapi32',
            'imports_has_crypto_apis', 'imports_has_network_apis', 'imports_has_process_apis',
            'exports_function_count', 'strings_count_total', 'strings_count_printable',
            'strings_count_unicode', 'strings_length_mean', 'strings_length_max',
            'strings_urls_count', 'strings_ips_count', 'strings_registry_keys',
            'strings_high_entropy_count', 'file_size', 'file_entropy',
            'file_size_virtual_ratio', 'timestamp_anomaly', 'timestamp_future',
            'timestamp_too_old', 'overlay_size', 'overlay_entropy', 'packing_ratio'
        ]
        
        for name in feature_names:
            defaults[name] = 0
        
        return defaults

    def _get_default_section_features(self):
        """Default section features"""
        return {
            'sections_executable_count': 0, 'sections_writable_count': 0,
            'sections_entropy_mean': 0, 'sections_entropy_std': 0,
            'sections_size_mean': 0, 'sections_size_std': 0,
            'sections_virtual_size_ratio': 0, 'sections_suspicious_names': 0,
            'sections_high_entropy_count': 0, 'sections_rwx_count': 0,
            'sections_zero_rawsize': 0, 'code_section_ratio': 0,
            'data_section_ratio': 0, 'resource_section_present': 0
        }

    def _get_default_string_features(self):
        """Default string features"""
        return {
            'strings_count_total': 0, 'strings_count_printable': 0,
            'strings_count_unicode': 0, 'strings_length_mean': 0,
            'strings_length_max': 0, 'strings_urls_count': 0,
            'strings_ips_count': 0, 'strings_registry_keys': 0,
            'strings_high_entropy_count': 0
        }

    def _get_default_metadata_features(self):
        """Default metadata features"""
        return {
            'file_size': 0, 'file_entropy': 0, 'file_size_virtual_ratio': 0,
            'timestamp_anomaly': 0, 'timestamp_future': 0, 'timestamp_too_old': 0,
            'overlay_size': 0, 'overlay_entropy': 0, 'packing_ratio': 1.0
        }

# Usage example
if __name__ == "__main__":
    extractor = PEFeatureExtractor()
    
    # Extract features from a PE file
    features = extractor.extract_features("sample.exe")
    
    # Convert to pandas DataFrame for ML pipeline
    df = pd.DataFrame([features])
    print(f"Extracted {len(features)} features")
    print(df.head())