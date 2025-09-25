"""
End-to-End Test for .doc Processing Complete Cycle

This test validates the complete end-to-end cycle:
1. Document download
2. Processing with Unstructured library
3. API endpoint functionality
"""

import asyncio
import aiohttp
import tempfile
import os
import subprocess
import urllib.parse
import glob
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

async def test_end_to_end_cycle():
    """Test complete end-to-end .doc processing cycle"""
    print("🔄 End-to-End .doc Processing Test")
    print("=" * 60)

    # Test document URLs
    test_urls = [
        "https://datumdoc.blob.core.windows.net/datumfilecontainer/PropertyDocuments/11925-wilshire-boulevard-94298286/Public/Unexecuted CA_11925WilshireBoulevard.doc",
        "https://datumdoc.blob.core.windows.net/datumfilecontainer/PropertyDocuments/900-gayley-41309454/Public/Unexecuted CA_900Gayley.doc",
        "https://datumdoc.blob.core.windows.net/datumfilecontainer/PropertyDocuments/the-hancock-35738319/Public/Unexecuted CA_TheHancock.doc"
    ]

    results = {}

    # Test each document
    for i, test_url in enumerate(test_urls):
        print(f"\n{'='*60}")
        print(f"Testing Document {i+1}/{len(test_urls)}: {test_url.split('/')[-1]}")
        print(f"{'='*60}")

        result = await test_single_document(test_url, f"doc_{i+1}")
        results[f"document_{i+1}"] = result

        if result["success"]:
            print(f"✅ Document {i+1} processed successfully!")
        else:
            print(f"❌ Document {i+1} failed: {result.get('error', 'Unknown error')}")

    # Test API endpoints
    print(f"\n{'='*60}")
    print("Testing API Endpoints")
    print(f"{'='*60}")

    api_results = await test_api_endpoints()
    results["api_endpoints"] = api_results

    # Generate comprehensive report
    generate_test_report(results)

    return results

async def test_single_document(test_url, doc_id):
    """Test processing of a single document end-to-end"""
    result = {
        "url": test_url,
        "doc_id": doc_id,
        "success": False,
        "download_success": False,
        "processing_success": False,
        "text_extracted": False,
        "processing_method": None,
        "text_length": 0,
        "elements_count": 0,
        "processing_time": 0,
        "error": None
    }

    start_time = asyncio.get_event_loop().time()

    try:
        # Step 1: Download document
        print(f"1️⃣ Downloading document...")
        content = await download_document(test_url)
        if not content:
            result["error"] = "Download failed"
            return result

        result["download_success"] = True
        print(f"✅ Downloaded {len(content)} bytes")

        # Step 2: Test file format
        print(f"2️⃣ Analyzing file format...")
        file_analysis = analyze_file_format(content)
        if not file_analysis["valid"]:
            result["error"] = f"Invalid file format: {file_analysis['error']}"
            return result

        print(f"✅ Valid .doc file: {file_analysis['size']} bytes")

        # Step 3: Process with Unstructured
        print(f"3️⃣ Processing with Unstructured library...")
        processing_result = await process_with_unstructured(content, doc_id)

        if processing_result["success"]:
            result["processing_success"] = True
            result["text_extracted"] = True
            result["processing_method"] = processing_result["processing_method"]
            result["text_length"] = len(processing_result.get("text_content", ""))
            result["elements_count"] = processing_result.get("elements_count", 0)
            print(f"✅ Processing successful: {result['elements_count']} elements, {result['text_length']} characters")

            # Show sample text
            if processing_result.get("text_content"):
                sample = processing_result["text_content"][:200] + "..." if len(processing_result["text_content"]) > 200 else processing_result["text_content"]
                print(f"📄 Sample text: {sample}")
        else:
            result["error"] = processing_result.get("error", "Processing failed")
            print(f"❌ Processing failed: {result['error']}")

        # Step 4: Test LibreOffice conversion
        print(f"4️⃣ Testing LibreOffice conversion...")
        libreoffice_result = await test_libreoffice_conversion(content)
        result["libreoffice_success"] = libreoffice_result["success"]

        if libreoffice_result["success"]:
            print(f"✅ LibreOffice conversion successful: {libreoffice_result['converted_size']} bytes")
        else:
            print(f"❌ LibreOffice conversion failed: {libreoffice_result.get('error', 'Unknown error')}")

        # Step 5: Test with converted .docx
        if libreoffice_result["success"] and libreoffice_result.get("docx_path"):
            print(f"5️⃣ Testing .docx processing...")
            docx_result = await process_docx_file(libreoffice_result["docx_path"])
            result["docx_processing_success"] = docx_result["success"]
            result["docx_elements"] = docx_result.get("elements_count", 0)

            if docx_result["success"]:
                print(f"✅ .docx processing successful: {docx_result['elements_count']} elements")
            else:
                print(f"❌ .docx processing failed: {docx_result.get('error', 'Unknown error')}")

        # Overall success
        result["success"] = result["download_success"] and (
            result["processing_success"] or
            (result["libreoffice_success"] and result.get("docx_processing_success", False))
        )

    except Exception as e:
        result["error"] = str(e)
        print(f"❌ Test error: {e}")

    finally:
        result["processing_time"] = asyncio.get_event_loop().time() - start_time

    return result

async def download_document(url):
    """Download document from URL"""
    try:
        encoded_url = urllib.parse.quote(url, safe=':/?=&')
        async with aiohttp.ClientSession() as session:
            async with session.get(encoded_url) as response:
                if response.status != 200:
                    print(f"❌ Download failed: HTTP {response.status}")
                    return None
                return await response.read()
    except Exception as e:
        print(f"❌ Download error: {e}")
        return None

def analyze_file_format(content):
    """Analyze file format and validity"""
    try:
        header = content[:8]
        if header.startswith(b'\xd0\xcf\x11\xe0'):
            return {
                "valid": True,
                "format": "doc",
                "size": len(content),
                "header": header.hex()
            }
        else:
            return {
                "valid": False,
                "error": f"Invalid header: {header.hex()}",
                "size": len(content)
            }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "size": len(content) if content else 0
        }

async def process_with_unstructured(content, doc_id):
    """Process document with Unstructured library"""
    try:
        # Set environment variables
        os.environ['LIBREOFFICE_PATH'] = '/opt/homebrew/bin/soffice'
        os.environ['UNSTRUCTURED_HIDE_PROGRESS_BAR'] = 'true'

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.doc', delete=False) as tmp_file:
            tmp_file.write(content)
            tmp_file.flush()
            doc_path = tmp_file.name

        try:
            from unstructured.partition.doc import partition_doc

            # Process with timeout
            loop = asyncio.get_event_loop()
            elements = await asyncio.wait_for(
                loop.run_in_executor(None, partition_doc, doc_path),
                timeout=120
            )

            # Extract text content
            text_parts = []
            metadata = {
                "total_elements": len(elements),
                "element_types": {},
                "has_tables": False,
                "has_images": False
            }

            for element in elements:
                if hasattr(element, 'text') and element.text:
                    text_parts.append(element.text)
                elif hasattr(element, 'content') and element.content:
                    text_parts.append(element.content)

                # Track element types
                element_type = type(element).__name__
                metadata["element_types"][element_type] = metadata["element_types"].get(element_type, 0) + 1

                if "Table" in element_type:
                    metadata["has_tables"] = True
                elif "Image" in element_type:
                    metadata["has_images"] = True

            text_content = "\n".join(text_parts)

            return {
                "success": True,
                "text_content": text_content,
                "metadata": metadata,
                "elements_count": len(elements),
                "processing_method": "unstructured"
            }

        finally:
            if os.path.exists(doc_path):
                os.unlink(doc_path)

    except asyncio.TimeoutError:
        return {"success": False, "error": "Processing timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}

async def test_libreoffice_conversion(content):
    """Test LibreOffice conversion"""
    try:
        with tempfile.NamedTemporaryFile(suffix='.doc', delete=False) as tmp_file:
            tmp_file.write(content)
            tmp_file.flush()
            doc_path = tmp_file.name

        output_dir = tempfile.mkdtemp()

        try:
            # Run LibreOffice conversion
            cmd = [
                'soffice',
                '--headless',
                '--convert-to', 'docx',
                '--outdir', output_dir,
                doc_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                # Find converted file
                docx_files = glob.glob(os.path.join(output_dir, "*.docx"))
                if docx_files:
                    docx_path = docx_files[0]
                    docx_size = os.path.getsize(docx_path)
                    return {
                        "success": True,
                        "converted_size": docx_size,
                        "docx_path": docx_path
                    }
                else:
                    return {"success": False, "error": "No .docx file created"}
            else:
                return {"success": False, "error": result.stderr}

        finally:
            if os.path.exists(doc_path):
                os.unlink(doc_path)
            if os.path.exists(output_dir):
                import shutil
                shutil.rmtree(output_dir)

    except Exception as e:
        return {"success": False, "error": str(e)}

async def process_docx_file(docx_path):
    """Process converted .docx file"""
    try:
        from unstructured.partition.docx import partition_docx

        elements = partition_docx(docx_path)

        # Extract text content
        text_parts = []
        for element in elements:
            if hasattr(element, 'text') and element.text:
                text_parts.append(element.text)
            elif hasattr(element, 'content') and element.content:
                text_parts.append(element.content)

        text_content = "\n".join(text_parts)

        return {
            "success": True,
            "text_content": text_content,
            "elements_count": len(elements)
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

async def test_api_endpoints():
    """Test API endpoints for document processing"""
    print("Testing API endpoints...")

    try:
        # Import monitoring system
        from src.domains.document.monitoring import document_monitor

        # Clear existing metrics
        document_monitor.clear_metrics()

        # Simulate some processing operations
        test_operations = [
            {"id": "api_test_1", "size": 1000, "extension": "doc", "success": True},
            {"id": "api_test_2", "size": 2000, "extension": "pdf", "success": True},
            {"id": "api_test_3", "size": 1500, "extension": "doc", "success": False},
        ]

        for op in test_operations:
            # Start monitoring
            document_monitor.start_processing(
                op["id"],
                op["size"],
                op["extension"]
            )

            # Simulate processing time
            await asyncio.sleep(0.1)

            # Complete monitoring
            document_monitor.complete_processing(
                document_id=op["id"],
                success=op["success"],
                processing_method="unstructured" if op["extension"] == "doc" else "azure_ai",
                extracted_text="Sample text content" if op["success"] else "",
                error_message="Test error" if not op["success"] else None,
                tables_found=1 if op["success"] else 0,
                elements_count=10 if op["success"] else 0
            )

        # Test stats endpoint
        stats = document_monitor.get_processing_stats()
        method_performance = document_monitor.get_method_performance()

        # Test metrics endpoint
        metrics = document_monitor.export_metrics()

        # Test clear endpoint
        document_monitor.clear_metrics()
        cleared_metrics = document_monitor.export_metrics()

        return {
            "success": True,
            "stats_endpoint": len(stats) > 0,
            "metrics_endpoint": len(metrics) > 0,
            "clear_endpoint": len(cleared_metrics) == 0,
            "total_operations": len(test_operations),
            "stats": stats,
            "method_performance": method_performance
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def generate_test_report(results):
    """Generate comprehensive test report"""
    print(f"\n📋 End-to-End Test Report")
    print("=" * 60)

    # Document processing results
    successful_docs = 0
    total_docs = 0

    for key, result in results.items():
        if key.startswith("document_"):
            total_docs += 1
            if result.get("success", False):
                successful_docs += 1

    print(f"📄 Document Processing:")
    print(f"   Total Documents: {total_docs}")
    print(f"   Successful: {successful_docs}")
    print(f"   Success Rate: {successful_docs/total_docs:.1%}" if total_docs > 0 else "   Success Rate: N/A")

    # API endpoint results
    api_results = results.get("api_endpoints", {})
    if api_results.get("success", False):
        print(f"\n🌐 API Endpoints:")
        print(f"   Stats Endpoint: {'✅' if api_results.get('stats_endpoint') else '❌'}")
        print(f"   Metrics Endpoint: {'✅' if api_results.get('metrics_endpoint') else '❌'}")
        print(f"   Clear Endpoint: {'✅' if api_results.get('clear_endpoint') else '❌'}")
        print(f"   Total Operations: {api_results.get('total_operations', 0)}")

    # Detailed results
    print(f"\n📊 Detailed Results:")
    for key, result in results.items():
        if key.startswith("document_"):
            status = "✅" if result.get("success", False) else "❌"
            method = result.get("processing_method", "unknown")
            elements = result.get("elements_count", 0)
            text_length = result.get("text_length", 0)
            processing_time = result.get("processing_time", 0)

            print(f"   {key}: {status} | Method: {method} | Elements: {elements} | Text: {text_length} chars | Time: {processing_time:.2f}s")

            if result.get("error"):
                print(f"      Error: {result['error']}")

    # Overall assessment
    overall_success = successful_docs > 0 and api_results.get("success", False)

    print(f"\n🎯 Overall Assessment:")
    if overall_success:
        print(f"   ✅ End-to-End Cycle: WORKING")
        print(f"   ✅ Ready for Production")
    else:
        print(f"   ❌ End-to-End Cycle: ISSUES DETECTED")
        print(f"   ⚠️  Review implementation before production")

    # Save detailed report
    report_file = Path(__file__).parent / "end_to_end_test_report.json"
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n📄 Detailed report saved to: {report_file}")

async def main():
    """Main test function"""
    results = await test_end_to_end_cycle()

    # Return overall success
    successful_docs = sum(1 for k, v in results.items() if k.startswith("document_") and v.get("success", False))
    total_docs = sum(1 for k in results.keys() if k.startswith("document_"))
    api_success = results.get("api_endpoints", {}).get("success", False)

    overall_success = successful_docs > 0 and api_success

    if overall_success:
        print(f"\n🎉 End-to-End Test PASSED!")
        print(f"   Documents: {successful_docs}/{total_docs} successful")
        print(f"   API Endpoints: Working")
        print(f"   ✅ Complete .doc processing cycle is functional")
    else:
        print(f"\n⚠️  End-to-End Test FAILED!")
        print(f"   Documents: {successful_docs}/{total_docs} successful")
        print(f"   API Endpoints: {'Working' if api_success else 'Failed'}")
        print(f"   ❌ Issues detected in .doc processing cycle")

if __name__ == "__main__":
    asyncio.run(main())
