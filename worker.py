import asyncio
import concurrent.futures
import time
import logging
import json
from math import floor
from multiprocessing import cpu_count
from typing import List, Optional, Literal, Dict, Tuple
from datetime import datetime
from pathlib import Path
import base64
from io import BytesIO
import functools
import os

import aiofiles
import boto3
from botocore.config import Config
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from ollama import AsyncClient
from pdf2image import convert_from_bytes
from PIL import Image

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Entity(BaseModel):
    type: str
    name: str
    confidence: Optional[float] = None
    attributes: Optional[dict] = None
    page_number: int
    position: Optional[dict] = None

class PageAnalysis(BaseModel):
    entities: List[Entity]
    text_content: Optional[str] = None
    page_number: int
    confidence: float

class DocumentAnalysis(BaseModel):
    document_id: str
    source: str
    total_pages: int
    pages: List[PageAnalysis]
    merged_entities: List[Entity]
    metadata: Dict = Field(default_factory=dict)

class ProcessingMetrics(BaseModel):
    duration: float
    success: bool
    error: Optional[str] = None
    cost: float = Field(default=0.0)
    entity_count: int = Field(default=0)
    page_count: int = Field(default=0)

class BatchReport(BaseModel):
    batch_id: str
    start_time: datetime
    end_time: datetime
    successful_count: int
    failed_count: int
    total_duration: float
    total_cost: float
    total_entities: int
    total_pages: int
    failed_documents: List[str]

class StorageHandler:
    def __init__(self):
        self.access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.endpoint_url = os.getenv('AWS_ENDPOINT_URL_S3')
        self.iam_endpoint = os.getenv('AWS_ENDPOINT_URL_IAM')
        self.region = os.getenv('AWS_REGION', 'auto')
        self.bucket = os.getenv('S3_BUCKET')
        self.prefix = os.getenv('S3_PREFIX', '')
        
        self.config = Config(
            retries={'max_attempts': 3},
            connect_timeout=30,
            read_timeout=60,
            region_name=self.region
        )
        
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        try:
            session = boto3.Session(
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key
            )
            
            return session.client(
                's3',
                config=self.config,
                endpoint_url=self.endpoint_url,
                verify=os.getenv('STORAGE_SSL_VERIFY', 'true').lower() == 'true'
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize storage client: {str(e)}")
            raise
    
    def get_document_batch(self, batch_size: int) -> List[dict]:
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=self.prefix,
                MaxKeys=batch_size
            )
            
            contents = response.get('Contents', [])
            # Filter out directories and zero-byte objects
            valid_documents = [
                doc for doc in contents 
                if doc.get('Size', 0) > 0 and not doc['Key'].endswith('/')
            ]
            
            if not valid_documents:
                logger.warning(f"No valid documents found in bucket {self.bucket} with prefix {self.prefix}")
            
            return valid_documents
            
        except Exception as e:
            logger.error(f"Failed to retrieve document batch: {str(e)}")
            raise
    
    def get_object(self, key: str) -> bytes:
        try:
            response = self.client.get_object(
                Bucket=self.bucket,
                Key=key
            )
            return response['Body'].read()
        except Exception as e:
            logger.error(f"Failed to retrieve object {key}: {str(e)}")
            raise

class ImageProcessor:
    @staticmethod
    def process_pdf_to_images(pdf_data: bytes) -> List[bytes]:
        images = convert_from_bytes(pdf_data)
        processed_images = []
        
        for image in images:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            processed_images.append(buffered.getvalue())
            
        return processed_images

    @staticmethod
    def optimize_image(image_data: bytes) -> bytes:
        image = Image.open(BytesIO(image_data))
        
        max_dimension = 2048
        if max(image.size) > max_dimension:
            ratio = max_dimension / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        buffered = BytesIO()
        image.save(buffered, format="PNG", optimize=True)
        return buffered.getvalue()

class DocumentProcessor:
    def __init__(self, batch_size: int = 50):
        self.storage_handler = StorageHandler()
        self.ollama_client = AsyncClient()
        self.batch_size = batch_size
        self.output_dir = Path(os.getenv('OUTPUT_DIR', './output'))
        self.output_dir.mkdir(exist_ok=True)
        
        self.max_workers = min(32, (cpu_count() or 1) * 4)
        self.process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers
        )
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers * 2
        )

    def fetch_document(self, document: dict) -> Tuple[str, bytes]:
        doc_key = document['Key']
        content = self.storage_handler.get_object(doc_key)
        return doc_key, content

    async def analyze_page(self, image_data: bytes, page_number: int) -> PageAnalysis:
        prompt = """
        Extract named entities from this document page for knowledge graph ingestion.
        Identify organizations, locations, people, dates, and key terms.
        Include position information where possible.
        """
        
        image_b64 = base64.b64encode(image_data).decode()
        
        try:
            response = await self.ollama_client.chat(
                model='llama3.2-vision',
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_b64]
                }],
                format=PageAnalysis.model_json_schema(),
                options={'temperature': 0}
            )
            
            page_analysis = PageAnalysis.model_validate_json(response.message.content)
            page_analysis.page_number = page_number
            return page_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing page {page_number}: {str(e)}")
            raise

    def merge_entities(self, pages: List[PageAnalysis]) -> List[Entity]:
        entity_map = {}
        
        for page in pages:
            for entity in page.entities:
                key = f"{entity.type}:{entity.name.lower()}"
                if key not in entity_map:
                    entity_map[key] = entity
                else:
                    existing = entity_map[key]
                    existing.attributes = existing.attributes or {}
                    existing.attributes['page_references'] = existing.attributes.get('page_references', [])
                    existing.attributes['page_references'].append(entity.page_number)
                    
                    if entity.confidence and (not existing.confidence or entity.confidence > existing.confidence):
                        existing.confidence = entity.confidence
        
        return list(entity_map.values())

    async def process_document(self, document: dict) -> ProcessingMetrics:
        start_time = time.time()
        try:
            if document['Size'] == 0:
                raise ValueError(f"Document {document['Key']} is empty")
                
            if not document['Key'].lower().endswith('.pdf'):
                raise ValueError(f"Document {document['Key']} is not a PDF file")
                
            doc_key, pdf_data = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                functools.partial(self.fetch_document, document)
            )
            
            if not pdf_data:
                raise ValueError(f"No data retrieved for document {doc_key}")
            
            raw_images = await asyncio.get_event_loop().run_in_executor(
                self.process_pool,
                functools.partial(ImageProcessor.process_pdf_to_images, pdf_data)
            )
            
            optimize_futures = [
                self.process_pool.submit(ImageProcessor.optimize_image, img_data)
                for img_data in raw_images
            ]
            
            optimized_images = []
            for future in concurrent.futures.as_completed(optimize_futures):
                optimized_images.append(future.result())
            
            page_tasks = [
                self.analyze_page(img_data, i+1)
                for i, img_data in enumerate(optimized_images)
            ]
            page_results = await asyncio.gather(*page_tasks)
            
            document_analysis = DocumentAnalysis(
                document_id=doc_key,
                source=f"s3://{self.storage_handler.bucket}/{doc_key}",
                total_pages=len(page_results),
                pages=page_results,
                merged_entities=self.merge_entities(page_results)
            )
            
            output_path = self.output_dir / f"{doc_key.replace('/', '_')}_analysis.json"
            async with aiofiles.open(output_path, mode='w') as f:
                await f.write(document_analysis.model_dump_json(indent=2))

            duration = time.time() - start_time
            return ProcessingMetrics(
                duration=duration,
                success=True,
                cost=0.006 * len(page_results),
                entity_count=len(document_analysis.merged_entities),
                page_count=len(page_results)
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error processing document {document['Key']}: {str(e)}")
            return ProcessingMetrics(
                duration=duration,
                success=False,
                error=str(e)
            )

    async def process_batch(self, batch_id: str, documents: List[dict]) -> BatchReport:
        start_time = datetime.now()
        tasks = [self.process_document(doc) for doc in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful = [r for r in results if isinstance(r, ProcessingMetrics) and r.success]
        failed = [r for r in results if isinstance(r, ProcessingMetrics) and not r.success]

        return BatchReport(
            batch_id=batch_id,
            start_time=start_time,
            end_time=datetime.now(),
            successful_count=len(successful),
            failed_count=len(failed),
            total_duration=sum(r.duration for r in results if isinstance(r, ProcessingMetrics)),
            total_cost=sum(r.cost for r in results if isinstance(r, ProcessingMetrics)),
            total_entities=sum(r.entity_count for r in results if isinstance(r, ProcessingMetrics)),
            total_pages=sum(r.page_count for r in results if isinstance(r, ProcessingMetrics)),
            failed_documents=[d['Key'] for d, r in zip(documents, results) 
                            if isinstance(r, ProcessingMetrics) and not r.success]
        )

    async def run(self):
        try:
            batch_id = 0
            reports = []
            
            while True:
                documents = self.storage_handler.get_document_batch(self.batch_size)
                if not documents:
                    break

                report = await self.process_batch(f"batch_{batch_id}", documents)
                reports.append(report)
                batch_id += 1

            await self.generate_final_report(reports)

        finally:
            self.process_pool.shutdown()
            self.thread_pool.shutdown()

    async def generate_final_report(self, batch_reports: List[BatchReport]):
        total_docs = sum(r.successful_count + r.failed_count for r in batch_reports)
        total_success = sum(r.successful_count for r in batch_reports)
        total_duration = sum(r.total_duration for r in batch_reports)
        total_cost = sum(r.total_cost for r in batch_reports)
        total_entities = sum(r.total_entities for r in batch_reports)
        total_pages = sum(r.total_pages for r in batch_reports)

        report = f"""Executive Summary - Document Processing Job
        * **Job Scope**: Processed {total_docs} documents ({total_pages} pages) for entity extraction
        * **Entity Extraction Results**:
        * Total Entities Extracted: {total_entities}
        * Average Entities per Document: {total_entities/total_success if total_success > 0 else 0:.1f}
        * Average Entities per Page: {total_entities/total_pages if total_pages > 0 else 0:.1f}
        * **Key Performance Metrics**:
        * Success Rate: {(total_success/total_docs)*100:.1f}% ({total_success} successful / {total_docs-total_success} failed)
        * Total Processing Duration: {total_duration/3600:.2f} hours
        * Total Cost: ${total_cost:.2f}
        * **Efficiency Metrics**:
        * Average Processing Speed: {total_duration/total_pages:.2f} seconds per page
        * Batch Processing Rate: {total_duration/len(batch_reports):.2f} seconds per batch
        * Cost per Entity: ${total_cost/total_entities if total_entities > 0 else 0:.4f}
        * **Infrastructure Used**:
        * CPU Cores: {cpu_count()}
        * Thread Pool Workers: {self.max_workers * 2}
        * Process Pool Workers: {self.max_workers}
        * Memory: {os.getenv('MEMORY_LIMIT', 'Unlimited')}
        * Model: {os.getenv('OLLAMA_MODEL', 'llama3.2-vision')}
        """

        async with aiofiles.open(self.output_dir / "job_report.md", mode='w') as f:
            await f.write(report)
async def main():
    processor = DocumentProcessor(
        batch_size=int(os.getenv('BATCH_SIZE', 2))
    )
    await processor.run()

if __name__ == "__main__":
    asyncio.run(main())