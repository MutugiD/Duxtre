"""
This test validates the RAG end-to-end cycle:
1. RAG files
2. Processing wof RAG
3. API endpoint functionality
"""

import json
from typing import Any, Optional, List, Dict
from datetime import datetime
import aiohttp
import os
import re
import pandas as pd

from src.core.settings import settings
import aiofiles

import src.domains.rag.schemas as rag_schemas
import src.domains.ai.flows as ai_flows
import src.domains.chat.schemas as chat_schemas
import src.infra.azure.manager as azure_manager
import src.domains.document.flows as document_processor

from loguru import logger


def convert_to_bool(value: Any) -> Optional[bool]:
    """
    Convert a value to a boolean.
    Returns None for empty strings, None values, or invalid conversions.
    """
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in ["true", "1", "yes", "on"]:
            return True
        elif value.lower() in ["false", "0", "no", "off"]:
            return False
    try:
        return bool(int(value))
    except (ValueError, TypeError):
        return None


def safe_int_conversion(value: Any) -> Optional[int]:
    """
    Safely convert a value to an integer.
    Returns None for empty strings, None values, or invalid conversions.
    """
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def convert_to_int(value: Any) -> Optional[int]:
    """
    Convert a value to an integer.
    Returns None for empty strings, None values, or invalid conversions.
    """
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def format_date_for_azure(date_str: Any) -> Optional[str]:
    """
    Format a date string for Azure AI Search DateTimeOffset.
    Returns None if the date is invalid or empty.
    """
    if not date_str or date_str == "":
        return None

    # If it's already a datetime object, format it properly for Azure
    if hasattr(date_str, "isoformat"):
        # Ensure timezone info is included for Azure DateTimeOffset
        if date_str.tzinfo is None:
            # Assume UTC if no timezone info
            from datetime import timezone

            date_str = date_str.replace(tzinfo=timezone.utc)
        return date_str.isoformat()

    # Try to parse common date formats
    import datetime
    from datetime import timezone

    date_formats = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
    ]

    for fmt in date_formats:
        try:
            dt = datetime.datetime.strptime(str(date_str), fmt)
            # Add UTC timezone for Azure DateTimeOffset
            dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except ValueError:
            continue

    # If we can't parse it, return None
    return None


async def insert_user_data(users_data: List[Dict]) -> bool:
    """
    Create embeddings and insert user data in Azure AI Search - individual embeddings per user
    :param users_data: List of user data dictionaries
    :return: True if the operation was successful, False otherwise
    """
    try:
        # Initialize Azure AI Search with force_recreate=False to preserve existing data
        await azure_manager.initialize_search(force_recreate=False)

        logger.info(
            f"Starting processing of {len(users_data)} users with individual embeddings"
        )

        # Process users in chunks for bulk upload (but individual embeddings)
        CHUNK_SIZE = 10  # Upload 10 users at a time
        total_chunks = (len(users_data) + CHUNK_SIZE - 1) // CHUNK_SIZE

        for chunk_index in range(0, len(users_data), CHUNK_SIZE):
            chunk = users_data[chunk_index : chunk_index + CHUNK_SIZE]
            chunk_num = (chunk_index // CHUNK_SIZE) + 1

            logger.info(
                f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} users)"
            )

            documents_to_upload = []

            # Process each user individually (like properties)
            for user_item in chunk:
                # Create simplified text for embedding (not massive JSON)
                user_text = f"User: {user_item.get('FirstName', '')} {user_item.get('LastName', '')} Company: {user_item.get('CompanyName', '')} Role: {user_item.get('Role', '')} Email: {user_item.get('Email', '')}"

                # Check if user has executed CA
                contact_activities_raw = user_item.get("ContactActivities", "[]")
                has_ca = False
                ca_date = None
                if contact_activities_raw:
                    try:
                        # Parse the JSON string
                        contact_activities = (
                            json.loads(contact_activities_raw)
                            if isinstance(contact_activities_raw, str)
                            else contact_activities_raw
                        )
                        for activity in contact_activities:
                            if "Executed CA" in activity.get("ActivityName", ""):
                                has_ca = True
                                ca_date = activity.get("CreatedDate")
                                user_text += " Has executed CA/NDA"
                                break
                    except (json.JSONDecodeError, TypeError):
                        # If parsing fails, assume no CA
                        pass

                # Generate individual embedding for this user
                embeddings = await ai_flows.generate_embeddings(user_text)

                # Create the document structure
                user_doc = {
                    "UserId": str(user_item.get("UserId"))
                    if user_item.get("UserId") not in [None, ""]
                    else f"user_{hash(json.dumps(user_item))}",
                    "Name": f"{user_item.get('FirstName', '')} {user_item.get('LastName', '')}".strip()
                    if user_item.get("FirstName") or user_item.get("LastName")
                    else user_item.get("Name"),
                    "Email": user_item.get("Email"),
                    "Phone": str(user_item.get("WorkPhone"))
                    if user_item.get("WorkPhone")
                    else str(user_item.get("MobilePhone"))
                    if user_item.get("MobilePhone")
                    else user_item.get("Phone"),
                    "Company": str(user_item.get("CompanyName"))
                    if user_item.get("CompanyName")
                    else str(user_item.get("Company"))
                    if user_item.get("Company")
                    else None,
                    "Title": user_item.get("Title"),
                    "SecurityRole": user_item.get("Role"),
                    # Additional user fields from actual data structure
                    "FirstName": user_item.get("FirstName"),
                    "LastName": user_item.get("LastName"),
                    "CompanyId": str(user_item.get("CompanyId"))
                    if user_item.get("CompanyId")
                    else None,
                    "CompanyUrl": user_item.get("CompanyUrl"),
                    "Address": user_item.get("Address"),
                    "City": user_item.get("City"),
                    "State": user_item.get("State"),
                    "ZipCode": str(user_item.get("ZipCode"))
                    if user_item.get("ZipCode") not in [None, ""]
                    else None,
                    "Country": user_item.get("Country"),
                    "WorkPhone": str(user_item.get("WorkPhone"))
                    if user_item.get("WorkPhone")
                    else None,
                    "MobilePhone": str(user_item.get("MobilePhone"))
                    if user_item.get("MobilePhone")
                    else None,
                    "IndustryRoleId": str(user_item.get("IndustryRoleId"))
                    if user_item.get("IndustryRoleId")
                    else None,
                    "InvestorType": user_item.get("InvestorType"),
                    "BrokerType": user_item.get("BrokerType"),
                    "UserType": user_item.get("UserType"),
                    # CA related fields
                    "IsCASigned": has_ca,
                    "NDASignedDateTime": format_date_for_azure(ca_date)
                    if ca_date
                    else None,
                    # OM related fields
                    "IsOMApproved": False,
                    "IsOMRequested": False,
                    "IsOMRejected": False,
                    "HasOMDefaultAccess": False,
                    "OMStatus": None,
                    "OMRequestedDate": None,
                    "OMApprovedDate": None,
                    "OMApprovedBy": None,
                    "OMRejectedDate": None,
                    "OMRejectedBy": None,
                    # DD related fields
                    "IsDDApproved": False,
                    "IsDDRequested": False,
                    "IsDDRejected": False,
                    "DDStatus": None,
                    "DuediligenceRequestDateTime": None,
                    "DuediligenceApprovedDateTime": None,
                    "DuediligenceApprovedBy": None,
                    "DuediligenceRejectDateTime": None,
                    "DuediligenceRejectBy": None,
                    # Associated property
                    "PropertyId": str(user_item.get("PropertyId", "")),
                    # Metadata
                    "CreatedAt": format_date_for_azure(datetime.now()),
                    "UpdatedAt": None,
                    # Raw data and individual embeddings
                    "raw_data": json.dumps(user_item),
                    "embeddings": embeddings,
                }

                documents_to_upload.append(user_doc)
                logger.info(f"Processed user {user_doc['Name']} (CA: {has_ca})")

            # Bulk upload this chunk
            logger.info(f"Uploading chunk {chunk_num} to Azure AI Search")
            try:
                success = await azure_manager.upload_user_documents_bulk(
                    documents_to_upload
                )
                if not success:
                    logger.error(f"Failed to upload chunk {chunk_num}")
                    return False
            except Exception as upload_error:
                logger.error(f"Exception during Azure upload: {str(upload_error)}")
                import traceback

                logger.error(f"Upload traceback: {traceback.format_exc()}")
                return False

            logger.info(f"Successfully uploaded chunk {chunk_num}/{total_chunks}")

        logger.info(f"Successfully processed all {len(users_data)} users")
        return True
    except Exception as e:
        logger.error(f"Error saving user records to Azure AI Search: {str(e)}")
        return False


async def detect_and_process_user_data(file_data: dict) -> bool:
    """
    Detect if the uploaded file contains user data and process it accordingly
    :param file_data: The uploaded file data
    :return: True if user data was found and processed, False otherwise
    """
    try:
        logger.info(f"Starting user data detection. File data type: {type(file_data)}")

        # Check if this looks like user data
        # Look for user-specific fields in the data - updated for actual data structure
        user_indicators = [
            "UserId",
            "FirstName",
            "LastName",
            "Role",
            "CompanyName",
            "Email",
            "IndustryRoleId",
            "UserType",
        ]

        # If file_data is a list, check the first item
        sample_data = (
            file_data[0]
            if isinstance(file_data, list) and len(file_data) > 0
            else file_data
        )

        logger.info(f"Sample data type: {type(sample_data)}")
        if isinstance(sample_data, dict):
            logger.info(
                f"Sample data keys: {list(sample_data.keys())[:10]}"
            )  # Show first 10 keys

            # Count how many user indicators are present
            user_field_count = sum(
                1 for field in user_indicators if field in sample_data
            )
            logger.info(
                f"Found {user_field_count} user indicators out of {len(user_indicators)}"
            )

            # If we have 3 or more user indicators, treat as user data
            if user_field_count >= 3:
                logger.info(
                    f"Detected user data with {user_field_count} user indicators"
                )

                # Process as user data
                if isinstance(file_data, list):
                    result = await insert_user_data(file_data)
                    logger.info(f"User data processing result: {result}")
                    return result
                else:
                    result = await insert_user_data([file_data])
                    logger.info(f"User data processing result: {result}")
                    return result
            else:
                logger.info(
                    f"Not enough user indicators ({user_field_count} < 3), treating as property data"
                )
        else:
            logger.info(f"Sample data is not a dict: {type(sample_data)}")

        return False
    except Exception as e:
        logger.error(f"Error detecting/processing user data: {str(e)}")
        import traceback

        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False


async def insert_file(context: rag_schemas.RagFileIn) -> bool:
    """
    Create embeddings and insert data in Azure AI Search
    Automatically detects whether the data is property or user data
    :param context: The RAG file input containing the file data
    :return: True if the operation was successful, False otherwise
    """
    try:
        logger.info("=== INSERT_FILE CALLED ===")
        logger.info(f"File JSON type: {type(context.file_json)}")
        logger.info(
            f"File JSON length: {len(context.file_json) if isinstance(context.file_json, list) else 'not a list'}"
        )

        # Initialize Azure AI Search with force_recreate=False to preserve existing data
        await azure_manager.initialize_search(force_recreate=False)

        logger.info("=== CALLING USER DETECTION ===")
        # Try to detect and process user data first
        user_data_processed = await detect_and_process_user_data(context.file_json)

        logger.info(f"=== USER DETECTION RESULT: {user_data_processed} ===")

        if user_data_processed:
            logger.info("Successfully processed file as user data")
            return True

        # If not user data, process as property data (existing logic)
        logger.info("Processing file as property data")

        # Process properties in chunks for bulk upload (like users)
        PROPERTY_CHUNK_SIZE = 10  # Upload 10 properties at a time
        total_chunks = (
            len(context.file_json) + PROPERTY_CHUNK_SIZE - 1
        ) // PROPERTY_CHUNK_SIZE

        logger.info(
            f"Starting processing of {len(context.file_json)} properties in {total_chunks} chunks"
        )

        successful_uploads = 0
        failed_uploads = 0

        for chunk_index in range(0, len(context.file_json), PROPERTY_CHUNK_SIZE):
            chunk = context.file_json[chunk_index : chunk_index + PROPERTY_CHUNK_SIZE]
            chunk_num = (chunk_index // PROPERTY_CHUNK_SIZE) + 1

            logger.info(
                f"Processing property chunk {chunk_num}/{total_chunks} ({len(chunk)} properties)"
            )

            documents_to_upload = []

            # Process each property individually for embeddings
            for item in chunk:
                try:
                    # Generate embeddings for the item
                    embeddings = await ai_flows.generate_embeddings(json.dumps(item))

                    # Create a document for Azure AI Search
                    document = {
                        "PropertyId": str(item.get("PropertyId"))
                        if item.get("PropertyId") not in [None, ""]
                        else f"prop_{hash(json.dumps(item))}",
                        "Address": None
                        if item.get("Address") == ""
                        else item.get("Address"),
                        "Address2": None
                        if item.get("Address2") == ""
                        else item.get("Address2"),
                        "City": None if item.get("City") == "" else item.get("City"),
                        "State": None if item.get("State") == "" else item.get("State"),
                        "ZipCode": str(item.get("ZipCode"))
                        if item.get("ZipCode") not in [None, ""]
                        else None,
                        "Country": None
                        if item.get("Country") == ""
                        else item.get("Country"),
                        "Latitude": None
                        if item.get("Latitude") == ""
                        else item.get("Latitude"),
                        "Longitude": None
                        if item.get("Longitude") == ""
                        else item.get("Longitude"),
                        "ListingName": None
                        if item.get("ListingName") == ""
                        else item.get("ListingName"),
                        "PropertyStage": None
                        if item.get("PropertyStage") == ""
                        else item.get("PropertyStage"),
                        "PropertyStatus": None
                        if item.get("PropertyStatus") == ""
                        else item.get("PropertyStatus"),
                        "IsFeatureListing": convert_to_bool(
                            item.get("IsFeatureListing")
                        ),
                        "FeatureListingDisplayOrder": safe_int_conversion(
                            item.get("FeatureListingDisplayOrder")
                        ),
                        "SellerContactName": None
                        if item.get("SellerContactName") == ""
                        else item.get("SellerContactName"),
                        "SellerCompanyName": None
                        if item.get("SellerCompanyName") == ""
                        else item.get("SellerCompanyName"),
                        "ListingExpiration": format_date_for_azure(
                            item.get("ListingExpiration")
                        ),
                        "PropertyType": None
                        if item.get("PropertyType") == ""
                        else item.get("PropertyType"),
                        "SquareFeet": safe_int_conversion(item.get("SquareFeet")),
                        "Units": safe_int_conversion(item.get("Units")),
                        "Occupancy": None
                        if item.get("Occupancy") == ""
                        else item.get("Occupancy"),
                        "YearBuilt": safe_int_conversion(item.get("YearBuilt")),
                        "YearRenovated": safe_int_conversion(item.get("YearRenovated")),
                        "CapitalInvested": None
                        if item.get("CapitalInvested") == ""
                        else item.get("CapitalInvested"),
                        "BuildingClass": None
                        if item.get("BuildingClass") == ""
                        else item.get("BuildingClass"),
                        "Building": None
                        if item.get("Building") == ""
                        else item.get("Building"),
                        "Stories": safe_int_conversion(item.get("Stories")),
                        "FloorPlates": None
                        if item.get("FloorPlates") == ""
                        else item.get("FloorPlates"),
                        "Elevators": safe_int_conversion(item.get("Elevators")),
                        "ParkingRatio": None
                        if item.get("ParkingRatio") == ""
                        else item.get("ParkingRatio"),
                        "Acres": None if item.get("Acres") == "" else item.get("Acres"),
                        "Zoning": None
                        if item.get("Zoning") == ""
                        else item.get("Zoning"),
                        "APN": None if item.get("APN") == "" else item.get("APN"),
                        "Tenancy": None
                        if item.get("Tenancy") == ""
                        else item.get("Tenancy"),
                        "LeaseType": None
                        if item.get("LeaseType") == ""
                        else item.get("LeaseType"),
                        "OwnershipInterest": None
                        if item.get("OwnershipInterest") == ""
                        else item.get("OwnershipInterest"),
                        "AskingPrice": None
                        if item.get("AskingPrice") == ""
                        else item.get("AskingPrice"),
                        "IsUnpriced": convert_to_bool(item.get("IsUnpriced")),
                        "PricePsf": None
                        if item.get("PricePsf") == ""
                        else item.get("PricePsf"),
                        "PricePerUnit": None
                        if item.get("Price/Unit") == ""
                        else item.get("Price/Unit"),
                        "PricePerAcre": None
                        if item.get("Price/Acre") == ""
                        else item.get("Price/Acre"),
                        "Walt": None if item.get("Walt") == "" else item.get("Walt"),
                        "CapRate": None
                        if item.get("CapRate") == ""
                        else item.get("CapRate"),
                        "NOI": None if item.get("NOI") == "" else item.get("NOI"),
                        "InPlaceCapRate": None
                        if item.get("In-PlaceCapRate") == ""
                        else item.get("In-PlaceCapRate"),
                        "InPlaceNOI": None
                        if item.get("In-PlaceNOI") == ""
                        else item.get("In-PlaceNOI"),
                        "T12Caprate": None
                        if item.get("T12Caprate") == ""
                        else item.get("T12Caprate"),
                        "T12Noi": None
                        if item.get("T12Noi") == ""
                        else item.get("T12Noi"),
                        "InPlaceRents": None
                        if item.get("InPlaceRents") == ""
                        else item.get("InPlaceRents"),
                        "MarketRents": None
                        if item.get("MarketRents") == ""
                        else item.get("MarketRents"),
                        "RentsBelowMarket": convert_to_bool(
                            item.get("RentsBelowMarket")
                        ),
                        "MarkToMarketCapRate": None
                        if item.get("Mark-to-MarketCapRate") == ""
                        else item.get("Mark-to-MarketCapRate"),
                        "Grm": None if item.get("Grm") == "" else item.get("Grm"),
                        "PotentialGrm": None
                        if item.get("PotentialGrm") == ""
                        else item.get("PotentialGrm"),
                        "InvestmentPeriod": None
                        if item.get("InvestmentPeriod") == ""
                        else item.get("InvestmentPeriod"),
                        "UnLeveredIrr": None
                        if item.get("UnLeveredIrr") == ""
                        else item.get("UnLeveredIrr"),
                        "LeveredIrr": None
                        if item.get("leveredIRR") == ""
                        else item.get("leveredIRR"),
                        "EquityMultiple": None
                        if item.get("EquityMultiple") == ""
                        else item.get("EquityMultiple"),
                        "CashOnCash": None
                        if item.get("CashOnCash") == ""
                        else item.get("CashOnCash"),
                        "ReturnOnCost": None
                        if item.get("ReturnOnCost") == ""
                        else item.get("ReturnOnCost"),
                        "DaysOnMarket": safe_int_conversion(item.get("DaysOnMarket")),
                        "ClosingCapRate": None
                        if item.get("ClosingCapRate") == ""
                        else item.get("ClosingCapRate"),
                        "SalesPrice": None
                        if item.get("SalesPrice") == ""
                        else item.get("SalesPrice"),
                        "SalePricePsf": None
                        if item.get("SalePricePsf") == ""
                        else item.get("SalePricePsf"),
                        "SpVsAp": None
                        if item.get("SpVsAp") == ""
                        else item.get("SpVsAp"),
                        "CloseSeller": None
                        if item.get("CloseSeller") == ""
                        else item.get("CloseSeller"),
                        "NumberOfTenants": safe_int_conversion(
                            item.get("NumberOfTenants")
                        ),
                        # Additional Zoning field (appears as Zoning.1 in CSV)
                        "ZoningAdditional": None
                        if item.get("Zoning.1") == ""
                        else item.get("Zoning.1"),
                        "FeaturedImage": None
                        if item.get("FeaturedImage") == ""
                        else item.get("FeaturedImage"),
                        "BannerImage": None
                        if item.get("BannerImage") == ""
                        else item.get("BannerImage"),
                        "PropertyTeamUrl": None
                        if item.get("PropertyTeamUrl") == ""
                        else item.get("PropertyTeamUrl"),
                        "IsCARequired": convert_to_bool(item.get("IsCARequired")),
                        "PageCount": safe_int_conversion(item.get("PageCount")),
                        "VisitorCount": safe_int_conversion(item.get("VisitorCount")),
                        "CASignedCount": safe_int_conversion(item.get("CASignedCount")),
                        "DownloadCount": safe_int_conversion(item.get("DownloadCount")),
                        "OfferCount": safe_int_conversion(item.get("OfferCount")),
                        # New fields from duxre_3001_property.csv
                        "DDRequestsCount": safe_int_conversion(
                            item.get("DDRequestsCount")
                        ),
                        "PropertyDescription": None
                        if item.get("PropertyDescription") == ""
                        else item.get("PropertyDescription"),
                        "MultiplePropertyImages": None
                        if item.get("MultiplePropertyImages") == ""
                        else item.get("MultiplePropertyImages"),
                        "MultipleOMDocuments": None
                        if item.get("MultipleOMDocuments") == ""
                        else item.get("MultipleOMDocuments"),
                        "MultipleDDDocuments": None
                        if item.get("MultipleDDDocuments") == ""
                        else item.get("MultipleDDDocuments"),
                        "CADocument": None
                        if item.get("CADocument") == ""
                        else item.get("CADocument"),
                        "raw_data": json.dumps(item),
                        "embeddings": embeddings,
                    }

                    documents_to_upload.append(document)
                    logger.info(f"Processed property {document['PropertyId']}")

                except Exception as item_error:
                    logger.error(
                        f"Failed to process property {item.get('PropertyId', 'unknown')}: {str(item_error)}"
                    )
                    failed_uploads += 1
                    continue

            # Bulk upload this chunk if we have any documents
            if documents_to_upload:
                logger.info(
                    f"Uploading property chunk {chunk_num} to Azure AI Search ({len(documents_to_upload)} documents)"
                )
                try:
                    success = await azure_manager.upload_documents_bulk(
                        documents_to_upload
                    )
                    if success:
                        successful_uploads += len(documents_to_upload)
                        logger.info(
                            f"Successfully uploaded property chunk {chunk_num}/{total_chunks}"
                        )
                    else:
                        logger.error(f"Failed to upload property chunk {chunk_num}")
                        failed_uploads += len(documents_to_upload)
                except Exception as upload_error:
                    logger.error(
                        f"Exception during property chunk upload: {str(upload_error)}"
                    )
                    import traceback

                    logger.error(f"Upload traceback: {traceback.format_exc()}")
                    failed_uploads += len(documents_to_upload)

        logger.info(
            f"Property processing complete. Successful: {successful_uploads}, Failed: {failed_uploads}"
        )

        # Return True if we uploaded at least some properties successfully
        return successful_uploads > 0

    except Exception as e:
        logger.error(f"Error saving property records to Azure AI Search: {str(e)}")
        return False


async def retrieve_relevant_chunks(context: chat_schemas.ChatIn) -> List[Dict]:
    """
    Retrieve relevant property records based on the user's query

    Args:
        context: The chat input containing the user's message

    Returns:
        A string containing the relevant property records as context for the LLM
    """
    try:
        # Generate embeddings for the query
        query_embeddings = await ai_flows.generate_embeddings(context.message)

        # Search for relevant documents using vector search
        results = await azure_manager.search_documents(
            query=context.message,
            vector=query_embeddings,
            top=20,
        )

        if not results:
            return []

        # Format the results as context for the LLM
        contexts = []
        for result in results:
            # Extract the most important fields for the context
            contexts.append(json.loads(result.get("raw_data")))

        return contexts
    except Exception as e:
        logger.error(f"Error retrieving relevant chunks: {str(e)}")
        return []


async def retrieve_relevant_user_chunks(context: chat_schemas.ChatIn) -> List[Dict]:
    """
    Retrieve relevant user records based on the user's query

    Args:
        context: The chat input containing the user's message

    Returns:
        A list containing the relevant user records as context for the LLM
    """
    try:
        # Generate embeddings for the query
        query_embeddings = await ai_flows.generate_embeddings(context.message)

        # Search for relevant user documents using vector search
        results = await azure_manager.search_user_documents(
            query=context.message,
            vector=query_embeddings,
            top=20,
        )

        if not results:
            return []

        # Format the results as context for the LLM
        contexts = []
        for result in results:
            # Extract the most important fields for the context
            contexts.append(json.loads(result.get("raw_data")))

        return contexts
    except Exception as e:
        logger.error(f"Error retrieving relevant user chunks: {str(e)}")
        return []


async def process_property_documents() -> bool:
    """
    Process all property documents from Azure Blob Storage:
    1. Load property data from Property.json
    2. For each property, extract document URLs
    3. Download and extract text from documents
    4. Create embeddings and upload to azure_property_document_index
    """
    try:
        # Initialize the azure_property_document_index
        await azure_manager.initialize_search(force_recreate=False)

        # Load property data
        with open("data/Property.json", "r") as f:
            properties_data = json.load(f)

        total_chunks_processed = 0

        for property_id, property_data in properties_data.items():
            logger.info(f"Processing documents for Property {property_id}")

            property_name = property_data.get("ListingName", f"Property {property_id}")

            # Extract document URLs from different fields
            document_urls = []

            # Offering Memorandum documents
            om_docs = property_data.get("MultipleOMDocuments", "")
            if om_docs and om_docs != "NaN":
                for url in om_docs.split(", "):
                    if url.strip():
                        document_urls.append(
                            {
                                "url": url.strip(),
                                "type": "Offering Memorandum",
                                "extension": url.split(".")[-1].lower()
                                if "." in url
                                else "unknown",
                            }
                        )

            # Due Diligence documents
            dd_docs = property_data.get("MultipleDDDocuments", "")
            if dd_docs and dd_docs != "NaN":
                for url in dd_docs.split(", "):
                    if url.strip():
                        document_urls.append(
                            {
                                "url": url.strip(),
                                "type": "Due Diligence",
                                "extension": url.split(".")[-1].lower()
                                if "." in url
                                else "unknown",
                            }
                        )

            # CA Documents
            ca_doc = property_data.get("CADocument", "")
            if ca_doc and ca_doc != "NaN":
                document_urls.append(
                    {
                        "url": ca_doc.strip(),
                        "type": "Confidentiality Agreement",
                        "extension": ca_doc.split(".")[-1].lower()
                        if "." in ca_doc
                        else "unknown",
                    }
                )

            # Process each document
            for doc_info in document_urls:
                try:
                    chunks = await extract_and_chunk_document(
                        doc_info["url"],
                        property_id,
                        property_name,
                        doc_info["type"],
                        doc_info["extension"],
                    )

                    if chunks:
                        # Upload chunks in batches
                        CHUNK_SIZE = 10
                        for i in range(0, len(chunks), CHUNK_SIZE):
                            chunk_batch = chunks[i : i + CHUNK_SIZE]
                            success = (
                                await azure_manager.upload_property_documents_bulk(
                                    chunk_batch
                                )
                            )
                            if success:
                                total_chunks_processed += len(chunk_batch)
                                logger.info(
                                    f"Uploaded {len(chunk_batch)} chunks for {doc_info['type']} of Property {property_id}"
                                )
                            else:
                                logger.error(
                                    f"Failed to upload chunks for {doc_info['type']} of Property {property_id}"
                                )

                except Exception as e:
                    logger.error(
                        f"Error processing document {doc_info['url']} for Property {property_id}: {str(e)}"
                    )
                    continue

        logger.info(
            f"Property document processing completed. Total chunks processed: {total_chunks_processed}"
        )
        return True

    except Exception as e:
        logger.error(f"Error in process_property_documents: {str(e)}")
        return False


async def extract_and_chunk_document(
    url: str, property_id: str, property_name: str, doc_type: str, team_id: Optional[str] = None, subscription_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Download a document from URL, extract text, chunk it, and create embeddings
    Enhanced with table extraction and CRE analysis (Phase 4.1)
    """

    try:
        # Extract file extension from URL, handling cases where URL has no extension
        if "." in url:
            parts = url.split(".")
            last_part = parts[-1].lower()
            # Check if the last part looks like a file extension (no slashes)

            if "/" not in last_part:
                file_extension = last_part
            else:
                file_extension = "unknown"
        else:
            file_extension = "unknown"

        logger.info(f"Processing document with extension: {file_extension}")

        # Download document - handle both HTTP URLs and local file paths
        if url.startswith('file://'):
            # Handle local file path
            file_path = url.replace('file://', '')
            if not os.path.exists(file_path):
                logger.error(f"Local file not found: {file_path}")
                return []

            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                logger.info(f"Read local file: {len(content)} bytes")
            except Exception as e:
                logger.error(f"Error reading local file {file_path}: {str(e)}")
                return []
        else:
            # Handle HTTP URLs
            async with aiohttp.ClientSession() as session:
                logger.info("Downloading document from URL")
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(
                            f"Failed to download document from {url}: HTTP {response.status}"
                        )
                        return []

                    content = await response.read()
                    logger.info(f"Downloaded {len(content)} bytes")

        logger.info(f"File extension detected: {file_extension}")

        # ROUTE to appropriate processor based on file type
        if file_extension.lower() == "doc":
            logger.info("Using Unstructured processor for .doc file")
            document_data = await document_processor.extract_document_advanced(
                content,
                file_extension,
                enable_table_enhancement=True
            )
        else:
            logger.info("Using Azure AI Document Intelligence for non-.doc file")
            document_data = await document_processor.extract_document_advanced(
                content,
                file_extension,
                enable_table_enhancement=True
            )

        if not document_data["text_content"].strip():
            logger.warning(f"No text extracted from {url}")
            return []

        # Extract text content and integrate table information
        from src.domains.document.utils import integrate_table_text_to_content
        text = integrate_table_text_to_content(document_data, document_data["text_content"])

        # Chunk the enhanced text (approximately 1000 characters per chunk)
        chunks = chunk_text(text, chunk_size=1000, overlap=100)

        # Create document objects with embeddings
        document_chunks = []
        document_name = url.split("/")[-1]

        for i, text_chunk in enumerate(chunks):
            try:
                # Generate embeddings for the chunk
                embeddings = await ai_flows.generate_embeddings(text_chunk)

                # Use URL + chunk index as ID
                url_clean = (
                    url.replace("https://", "")
                    .replace("http://", "")
                    .replace("/", "_")
                    .replace(".", "_")
                )
                # Replace spaces and other invalid characters with underscores
                url_clean = re.sub(r"[^a-zA-Z0-9_\-=]", "_", url_clean)
                chunk_id = f"{url_clean}_chunk_{i}"

                chunk_doc = {
                    "id": chunk_id,
                    "document_url": url,
                    "property_id": property_id,
                    "property_name": property_name,
                    "document_type": doc_type,
                    "document_name": document_name,
                    "chunk_text": text_chunk,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "file_extension": file_extension,
                    "created_date": datetime.utcnow().isoformat() + "Z",
                    "embeddings": embeddings,
                    "team_id": team_id,
                    "subscription_name": subscription_name,
                }
                document_chunks.append(chunk_doc)

            except Exception as chunk_error:
                logger.error(
                    f"Error processing chunk {i} for {url}: {str(chunk_error)}"
                )
                continue

        logger.info(
            f"Successfully created {len(document_chunks)} chunks from {url}"
        )
        return document_chunks

    except Exception as e:
        logger.error(f"Error in extract_and_chunk_document for {url}: {str(e)}")
        return []





def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence or word boundary
        if end < len(text):
            # Look for sentence ending
            sentence_end = text.rfind(".", start, end)
            if sentence_end > start + chunk_size // 2:
                end = sentence_end + 1
            else:
                # Look for word boundary
                word_end = text.rfind(" ", start, end)
                if word_end > start + chunk_size // 2:
                    end = word_end

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start >= len(text):
            break

    return chunks


async def process_property_documents_test() -> bool:
    """
    Test version of property document processing to debug issues
    """
    try:
        logger.info("Starting property documents test processing")

        # First, just test if we can create the index
        await azure_manager.initialize_search(force_recreate=False)
        logger.info("Azure search initialized successfully")

        # Test if we can load the Property.json file
        file_path = "data/Property.json"
        if not os.path.exists(file_path):
            logger.error(f"Property.json file not found at {file_path}")
            return False

        logger.info(f"Property.json file found at {file_path}")

        with open(file_path, "r") as f:
            properties_data = json.load(f)

        logger.info(f"Loaded {len(properties_data)} properties from Property.json")

        # Just process one property as a test
        first_property_id = list(properties_data.keys())[0]
        first_property = properties_data[first_property_id]

        logger.info(
            f"Testing with property {first_property_id}: {first_property.get('ListingName', 'Unknown')}"
        )

        # Check if this property has any documents
        om_docs = first_property.get("MultipleOMDocuments", "")
        ca_doc = first_property.get("CADocument", "")
        dd_docs = first_property.get("MultipleDDDocuments", "")

        logger.info(f"Property {first_property_id} documents:")
        logger.info(f"  OM Documents: {om_docs}")
        logger.info(f"  CA Document: {ca_doc}")
        logger.info(f"  DD Documents: {dd_docs}")

        return True

    except Exception as e:
        logger.error(f"Error in process_property_documents_test: {str(e)}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


async def process_property_documents_robust() -> bool:
    """
    Process property documents by extracting URLs using regex instead of JSON parsing
    This avoids JSON syntax issues and focuses on extracting and indexing document URLs
    """
    try:
        logger.info(
            "Starting robust property document processing with regex extraction"
        )

        # Initialize the azure_property_document_index
        await azure_manager.initialize_search(force_recreate=False)

        # Read the file as text and extract URLs using regex
        with open("data/Property.json", "r") as f:
            content = f.read()

        # Extract property IDs
        property_ids = re.findall(r'"(\d+)":\s*{', content)
        logger.info(f"Found {len(property_ids)} properties to process")

        # Extract property names for each property ID
        property_names = {}
        for prop_id in property_ids:
            # Look for ListingName near this property ID
            pattern = rf'"{prop_id}":\s*{{[^}}]*?"ListingName":\s*"([^"]*)"'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                property_names[prop_id] = match.group(1)
            else:
                property_names[prop_id] = f"Property {prop_id}"

        # Extract all document URLs with their associated property IDs
        document_patterns = [
            (r'"MultipleOMDocuments":\s*"([^"]*)"', "Offering Memorandum"),
            (r'"MultipleDDDocuments":\s*"([^"]*)"', "Due Diligence"),
            (r'"CADocument":\s*"([^"]*)"', "Confidentiality Agreement"),
            (r'"PropertyDocument":\s*\[\s*"([^"]*)"', "Property Document"),
        ]

        total_documents_processed = 0

        # For each property, find its document URLs
        for prop_id in property_ids:
            logger.info(
                f"Processing documents for Property {prop_id}: {property_names[prop_id]}"
            )

            # Find the property section in the JSON
            property_start = content.find(f'"{prop_id}": {{')
            if property_start == -1:
                continue

            # Find the end of this property section (next property or end of file)
            next_property_start = len(content)
            for other_id in property_ids:
                if other_id != prop_id:
                    other_start = content.find(f'"{other_id}": {{', property_start + 1)
                    if other_start != -1 and other_start < next_property_start:
                        next_property_start = other_start

            property_section = content[property_start:next_property_start]

            # Extract document URLs from this property section
            for pattern, doc_type in document_patterns:
                matches = re.findall(pattern, property_section, re.DOTALL)

                for match in matches:
                    if (
                        match
                        and match != "NaN"
                        and "datumdoc.blob.core.windows.net" in match
                    ):
                        # Handle comma-separated URLs
                        urls = [url.strip() for url in match.split(",") if url.strip()]

                        for url in urls:
                            if url and "datumdoc.blob.core.windows.net" in url:
                                try:
                                    logger.info(
                                        f"Processing document: {doc_type} for Property {prop_id}"
                                    )

                                    # Extract file extension to check if it's supported
                                    if "." in url:
                                        parts = url.split(".")
                                        last_part = parts[-1].lower()
                                        # Check if the last part looks like a file extension (no slashes)
                                        if "/" not in last_part:
                                            file_extension = last_part
                                        else:
                                            file_extension = "unknown"
                                    else:
                                        file_extension = "unknown"

                                    # Process only PDF and DOC files
                                    if file_extension in ["pdf", "doc", "docx"]:
                                        chunks = await extract_and_chunk_document(
                                            url,
                                            prop_id,
                                            property_names[prop_id],
                                            doc_type,
                                        )

                                        if chunks:
                                            # Upload chunks in batches
                                            CHUNK_SIZE = (
                                                5  # Smaller batch size for stability
                                            )
                                            for i in range(0, len(chunks), CHUNK_SIZE):
                                                chunk_batch = chunks[i : i + CHUNK_SIZE]
                                                success = await azure_manager.upload_property_documents_bulk(
                                                    chunk_batch
                                                )
                                                if success:
                                                    total_documents_processed += len(
                                                        chunk_batch
                                                    )
                                                    logger.info(
                                                        f"Uploaded {len(chunk_batch)} chunks for {doc_type} of Property {prop_id}"
                                                    )
                                                else:
                                                    logger.error(
                                                        f"Failed to upload chunks for {doc_type} of Property {prop_id}"
                                                    )
                                    else:
                                        logger.info(
                                            f"Skipping non-text file: {file_extension} for {url}"
                                        )

                                except Exception as e:
                                    logger.error(
                                        f"Error processing document {url} for Property {prop_id}: {str(e)}"
                                    )
                                    continue

        logger.info(
            f"Robust property document processing completed. Total chunks processed: {total_documents_processed}"
        )
        return True

    except Exception as e:
        logger.error(f"Error in process_property_documents_robust: {str(e)}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


async def process_property_documents_from_csv() -> bool:
    """
    Process property documents by reading URLs from Property.csv file
    This is the clean approach using CSV data instead of broken JSON
    """
    try:
        logger.info("Starting property document processing from CSV")

        # Initialize the azure_property_document_index
        await azure_manager.initialize_search(force_recreate=False)

        # Read the Property.csv file
        df = pd.read_csv("data/Property.csv")
        logger.info(f"Loaded {len(df)} properties from Property.csv")

        total_documents_processed = 0
        successful_documents = 0
        failed_documents = 0

        # Process each property
        for _, row in df.iterrows():
            property_id = str(row["PropertyId"])
            property_name = row.get("ListingName", f"Property {property_id}")

            logger.info(
                f"Processing documents for Property {property_id}: {property_name}"
            )

            # Extract document URLs from different columns
            document_sources = [
                ("MultipleOMDocuments", "Offering Memorandum"),
                ("MultipleDDDocuments", "Due Diligence"),
                ("CADocument", "Confidentiality Agreement"),
            ]

            for column, doc_type in document_sources:
                if column in row and pd.notna(row[column]):
                    urls_text = str(row[column])

                    # Extract full URLs from the text - no capture groups
                    full_urls = re.findall(
                        r"https://datumdoc\.blob\.core\.windows\.net/[^\s,]*\.(?:pdf|doc|docx)",
                        urls_text,
                        re.IGNORECASE,
                    )

                    for url in full_urls:
                        try:
                            # Skip URLs that are likely to fail
                            if not url or "PropertyCADocuments" in url:
                                logger.info(f"Skipping CA document (often 404): {url}")
                                continue

                            logger.info(
                                f"Processing document: {doc_type} for Property {property_id} - URL: {url}"
                            )

                            # Extract file extension to check if it's supported
                            if "." in url:
                                parts = url.split(".")
                                last_part = parts[-1].lower()
                                # Check if the last part looks like a file extension (no slashes)
                                if "/" not in last_part:
                                    file_extension = last_part
                                else:
                                    file_extension = "unknown"
                            else:
                                file_extension = "unknown"

                            # Process only PDF and DOC files
                            if file_extension in ["pdf", "doc", "docx"]:
                                chunks = await extract_and_chunk_document(
                                    url,
                                    property_id,
                                    property_name,
                                    doc_type,
                                )

                                if chunks:
                                    # Upload chunks in batches
                                    CHUNK_SIZE = (
                                        3  # Smaller batch for better error handling
                                    )
                                    for i in range(0, len(chunks), CHUNK_SIZE):
                                        chunk_batch = chunks[i : i + CHUNK_SIZE]
                                        try:
                                            success = await azure_manager.upload_property_documents_bulk(
                                                chunk_batch
                                            )
                                            if success:
                                                total_documents_processed += len(
                                                    chunk_batch
                                                )
                                                successful_documents += 1
                                                logger.info(
                                                    f"✅ Uploaded {len(chunk_batch)} chunks for {doc_type} of Property {property_id}"
                                                )
                                            else:
                                                failed_documents += 1
                                                logger.error(
                                                    f"❌ Failed to upload chunks for {doc_type} of Property {property_id}"
                                                )
                                        except Exception as upload_error:
                                            failed_documents += 1
                                            logger.error(
                                                f"❌ Upload exception for {doc_type} of Property {property_id}: {str(upload_error)}"
                                            )
                                else:
                                    failed_documents += 1
                                    logger.warning(f"⚠️ No chunks extracted from {url}")
                            else:
                                logger.info(f"Skipping non-text file: {file_extension}")

                        except Exception as e:
                            failed_documents += 1
                            logger.error(
                                f"❌ Error processing document {url} for Property {property_id}: {str(e)}"
                            )
                            continue

        logger.info("CSV property document processing completed:")
        logger.info(f"  ✅ Total chunks processed: {total_documents_processed}")
        logger.info(f"  ✅ Successful documents: {successful_documents}")
        logger.info(f"  ❌ Failed documents: {failed_documents}")

        return total_documents_processed > 0

    except Exception as e:
        logger.error(f"Error in process_property_documents_from_csv: {str(e)}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


async def process_property_data_from_json(
    file_path: str = "data/Property.json",
) -> bool:
    """
    Reads property data from a JSON file, cleans it, and uploads it to Azure AI Search.

    Args:
        file_path: The path to the JSON file.

    Returns:
        True if processing was successful, False otherwise.
    """
    logger.info(f"Starting property data processing from JSON file: {file_path}")
    try:
        # Asynchronously read and clean the JSON file content
        async with aiofiles.open(file_path, "r") as f:
            json_content = await f.read()

        # In-memory replacement is fast, no need for async here
        cleaned_json_content = json_content.replace("NaN", "null")

        data = json.loads(cleaned_json_content)
        logger.info(
            f"Successfully loaded and cleaned JSON data. Found {len(data)} properties."
        )

        properties_to_upload = []
        for property_id, prop_details in data.items():
            # Create a new dictionary for the property to be uploaded
            property_doc = {"PropertyId": property_id}

            # Add all other details from the JSON object
            for key, value in prop_details.items():
                # Clean field name - replace dots and other invalid characters
                clean_key = key.replace(".", "_").replace("/", "_").replace(" ", "_")

                # Clean HTML tags from description
                if key == "PropertyDescription" and isinstance(value, str):
                    property_doc[clean_key] = html.unescape(
                        re.sub("<[^<]+?>", "", value)
                    )
                # Convert nested lists of dicts to JSON strings
                elif isinstance(value, list) and all(
                    isinstance(i, dict) for i in value
                ):
                    property_doc[clean_key] = json.dumps(value)
                # Handle boolean fields properly
                elif clean_key in [
                    "IsFeatureListing",
                    "IsUnpriced",
                    "RentsBelowMarket",
                    "IsCARequired",
                ]:
                    property_doc[clean_key] = (
                        bool(value) if value is not None else False
                    )
                # Handle date fields - convert NaT and invalid dates to null
                elif "Date" in clean_key or "Expiration" in clean_key:
                    if value is None or str(value) in ["NaT", "nat", ""]:
                        property_doc[clean_key] = None
                    else:
                        property_doc[clean_key] = str(value) if value != "NaT" else None
                # Handle numeric fields (Int32, Double) - convert empty strings to null
                elif clean_key in [
                    "SquareFeet",
                    "Units",
                    "YearBuilt",
                    "YearRenovated",
                    "FeatureListingDisplayOrder",
                    "DaysOnMarket",
                    "NumberOfTenants",
                    "PageCount",
                    "VisitorCount",
                    "CASignedCount",
                    "DownloadCount",
                    "OfferCount",
                    "DDRequestsCount",
                    "Stories",
                    "Elevators",
                ]:
                    if value is None or str(value) in ["", "NaN", "nan"]:
                        property_doc[clean_key] = None
                    else:
                        try:
                            property_doc[clean_key] = int(float(str(value)))
                        except (ValueError, TypeError):
                            property_doc[clean_key] = None
                # Handle double fields
                elif clean_key in [
                    "Latitude",
                    "Longitude",
                    "Occupancy",
                    "CapitalInvested",
                    "Acres",
                    "AskingPrice",
                    "PricePsf",
                    "PricePerUnit",
                    "PricePerAcre",
                    "Walt",
                    "CapRate",
                    "NOI",
                    "InPlaceCapRate",
                    "InPlaceNOI",
                    "T12Caprate",
                    "T12Noi",
                    "InPlaceRents",
                    "MarketRents",
                    "MarkToMarketCapRate",
                    "Grm",
                    "PotentialGrm",
                    "UnLeveredIrr",
                    "LeveredIrr",
                    "EquityMultiple",
                    "CashOnCash",
                    "ReturnOnCost",
                    "ClosingCapRate",
                    "SalesPrice",
                    "SalePricePsf",
                    "SpVsAp",
                ]:
                    if value is None or str(value) in ["", "NaN", "nan"]:
                        property_doc[clean_key] = None
                    else:
                        try:
                            property_doc[clean_key] = float(str(value))
                        except (ValueError, TypeError):
                            property_doc[clean_key] = None
                # Convert all other values to strings, but handle empty values properly
                else:
                    if value is None or str(value) in ["", "NaN", "nan", "null"]:
                        property_doc[clean_key] = None
                    else:
                        property_doc[clean_key] = str(value)

            properties_to_upload.append(property_doc)

        if not properties_to_upload:
            logger.warning("No properties found to upload.")
            return False

        logger.info(
            f"Preparing to upload {len(properties_to_upload)} properties to Azure AI Search."
        )

        # Upload to Azure AI Search in batches
        batch_size = 50
        for i in range(0, len(properties_to_upload), batch_size):
            batch = properties_to_upload[i : i + batch_size]
            success = await azure_manager.upload_documents_bulk(batch)
            if success:
                logger.info(
                    f"Successfully uploaded batch {i // batch_size + 1} of properties."
                )
            else:
                logger.error(
                    f"Failed to upload batch {i // batch_size + 1} of properties."
                )
                return False

        logger.info("Successfully uploaded all property documents from JSON.")
        return True

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}: {e}")
        return False
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return False
    except Exception as e:
        logger.error(
            f"An unexpected error occurred in process_property_data_from_json: {e}"
        )
        return False


async def process_users_data_from_json(file_path: str = "data/Users.json") -> bool:
    """
    Reads user data from a JSON file, processes it, and uploads it to Azure AI Search.
    This function processes the file line by line to handle large files.

    Args:
        file_path: The path to the Users.json file.

    Returns:
        True if processing was successful, False otherwise.
    """
    logger.info(f"Starting user data processing from JSON file: {file_path}")
    documents = []
    batch_size = 500
    try:
        async with aiofiles.open(file_path, "r") as f:
            async for line in f:
                # Skip empty lines
                if not line.strip():
                    continue

                try:
                    user_data = json.loads(line)
                    # Assuming 'UserId' is the unique identifier in your user data
                    user_data["id"] = str(user_data.get("UserId", ""))

                    # Convert all values to string to avoid type errors in search index
                    for key, value in user_data.items():
                        user_data[key] = str(value) if value is not None else ""

                    documents.append(user_data)

                    if len(documents) >= batch_size:
                        await azure_manager.upload_user_documents_bulk(documents)
                        logger.info(
                            f"Uploaded a batch of {len(documents)} user documents."
                        )
                        documents = []

                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON line: {line.strip()}")
                    continue

        # Upload any remaining documents
        if documents:
            await azure_manager.upload_user_documents_bulk(documents)
            logger.info(f"Uploaded the final batch of {len(documents)} user documents.")

        logger.info("Successfully processed and uploaded all user data from JSON.")
        return True

    except FileNotFoundError:
        logger.error(f"Users.json file not found at {file_path}")
        return False
    except Exception as e:
        logger.error(f"An error occurred during user data processing: {e}")
        return False


def clean_html(raw_html):
    # ... existing code ...
    return True


async def insert_property_data_simple(properties_data: List[Dict]) -> bool:
    """
    Insert property data into the simplified Azure AI Search index
    No type conversions - just raw JSON with embeddings

    Args:
        properties_data: List of property dictionaries

    Returns:
        True if successful, False otherwise
    """
    try:
        # Initialize the simplified search index
        await azure_manager.initialize_simple_search(force_recreate=False)

        logger.info(
            f"Starting processing of {len(properties_data)} properties with simplified schema"
        )

        # Process properties in chunks for bulk upload
        CHUNK_SIZE = 50
        for i in range(0, len(properties_data), CHUNK_SIZE):
            chunk = properties_data[i : i + CHUNK_SIZE]

            # Prepare documents for this chunk
            documents = []
            for idx, item in enumerate(chunk):
                try:
                    # Create a searchable content string from key fields
                    content_parts = []

                    # Add key property info to content
                    if item.get("ListingName"):
                        content_parts.append(f"Property: {item['ListingName']}")
                    if item.get("Address"):
                        content_parts.append(f"Address: {item['Address']}")
                    if item.get("City"):
                        content_parts.append(f"City: {item['City']}")
                    if item.get("State"):
                        content_parts.append(f"State: {item['State']}")
                    if item.get("PropertyType"):
                        content_parts.append(f"Type: {item['PropertyType']}")
                    if item.get("AskingPrice"):
                        content_parts.append(f"Price: ${item['AskingPrice']}")
                    if item.get("PropertyDescription"):
                        content_parts.append(
                            f"Description: {item['PropertyDescription']}"
                        )

                    content = " | ".join(content_parts)

                    # Generate embeddings for the content
                    embeddings = await ai_flows.generate_embeddings(content)

                    # Create document for Azure Search
                    document = {
                        "id": str(
                            item.get("PropertyId", f"prop_{i * CHUNK_SIZE + idx}")
                        ),
                        "content": content,
                        "raw_data": json.dumps(item),
                        "embeddings": embeddings,
                    }

                    documents.append(document)

                except Exception as e:
                    logger.error(f"Error processing property {idx}: {str(e)}")
                    continue

            # Upload documents in bulk
            if documents:
                success = await azure_manager.upload_simple_property_documents(
                    documents
                )
                if success:
                    logger.info(
                        f"Successfully uploaded {len(documents)} properties (chunk {i // CHUNK_SIZE + 1})"
                    )
                else:
                    logger.error(f"Failed to upload chunk {i // CHUNK_SIZE + 1}")
                    return False

        logger.info("Property data insertion completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error in insert_property_data_simple: {str(e)}")
        return False


async def insert_user_data_simple(users_data: List[Dict]) -> bool:
    """
    Insert user data into the simplified Azure AI Search index
    No type conversions - just raw JSON with embeddings

    Args:
        users_data: List of user dictionaries

    Returns:
        True if successful, False otherwise
    """
    try:
        # Initialize the simplified search index
        await azure_manager.initialize_simple_search(force_recreate=False)

        logger.info(
            f"Starting processing of {len(users_data)} users with simplified schema"
        )

        # Process users in chunks for bulk upload
        CHUNK_SIZE = 50
        for i in range(0, len(users_data), CHUNK_SIZE):
            chunk = users_data[i : i + CHUNK_SIZE]

            # Prepare documents for this chunk
            documents = []
            for idx, item in enumerate(chunk):
                try:
                    # Create a searchable content string from key fields
                    content_parts = []

                    # Add key user info to content
                    if item.get("FirstName") and item.get("LastName"):
                        content_parts.append(
                            f"Name: {item['FirstName']} {item['LastName']}"
                        )
                    if item.get("Email"):
                        content_parts.append(f"Email: {item['Email']}")
                    if item.get("Company"):
                        content_parts.append(f"Company: {item['Company']}")
                    if item.get("Title"):
                        content_parts.append(f"Title: {item['Title']}")
                    if item.get("City"):
                        content_parts.append(f"City: {item['City']}")
                    if item.get("State"):
                        content_parts.append(f"State: {item['State']}")
                    if item.get("InvestorType"):
                        content_parts.append(f"Investor Type: {item['InvestorType']}")
                    if item.get("SecurityRole"):
                        content_parts.append(f"Role: {item['SecurityRole']}")

                    content = " | ".join(content_parts)

                    # Generate embeddings for the content
                    embeddings = await ai_flows.generate_embeddings(content)

                    # Create document for Azure Search
                    document = {
                        "id": str(item.get("UserId", f"user_{i * CHUNK_SIZE + idx}")),
                        "content": content,
                        "raw_data": json.dumps(item),
                        "embeddings": embeddings,
                    }

                    documents.append(document)

                except Exception as e:
                    logger.error(f"Error processing user {idx}: {str(e)}")
                    continue

            # Upload documents in bulk
            if documents:
                success = await azure_manager.upload_simple_user_documents(documents)
                if success:
                    logger.info(
                        f"Successfully uploaded {len(documents)} users (chunk {i // CHUNK_SIZE + 1})"
                    )
                else:
                    logger.error(f"Failed to upload chunk {i // CHUNK_SIZE + 1}")
                    return False

        logger.info("User data insertion completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error in insert_user_data_simple: {str(e)}")
        return False


async def retrieve_relevant_chunks_simple(context: chat_schemas.ChatIn) -> List[Dict]:
    """
    Retrieve relevant property records from simplified index

    Args:
        context: The chat input containing the user's message

    Returns:
        A list containing the relevant property records as context for the LLM
    """
    try:
        # Generate embeddings for the query
        query_embeddings = await ai_flows.generate_embeddings(context.message)

        # Search for relevant documents using hybrid search
        results = await azure_manager.search_simple_property_documents(
            query=context.message,
            vector=query_embeddings,
            top=20,
        )

        if not results:
            return []

        # Extract raw data from results
        contexts = []
        for result in results:
            raw_data = result.get("raw_data")
            if raw_data:
                contexts.append(json.loads(raw_data))

        return contexts
    except Exception as e:
        logger.error(
            f"Error retrieving relevant chunks from simplified index: {str(e)}"
        )
        return []


async def retrieve_relevant_user_chunks_simple(
    context: chat_schemas.ChatIn,
) -> List[Dict]:
    """
    Retrieve relevant user records from simplified index

    Args:
        context: The chat input containing the user's message

    Returns:
        A list containing the relevant user records as context for the LLM
    """
    try:
        # Generate embeddings for the query
        query_embeddings = await ai_flows.generate_embeddings(context.message)

        # Search for relevant user documents using hybrid search
        results = await azure_manager.search_simple_user_documents(
            query=context.message,
            vector=query_embeddings,
            top=20,
        )

        if not results:
            return []

        # Extract raw data from results
        contexts = []
        for result in results:
            raw_data = result.get("raw_data")
            if raw_data:
                contexts.append(json.loads(raw_data))

        return contexts
    except Exception as e:
        logger.error(
            f"Error retrieving relevant user chunks from simplified index: {str(e)}"
        )
        return []


async def retrieve_property_documents(
    property_id: str, query: str, document_type: str = None
) -> List[Dict]:
    """
    Search for documents specific to a property
    Enables queries like: "what is the NDA detail of Ocean Park Plaza property?"

    Args:
        property_id: The property ID to search documents for
        query: The search query (e.g., "NDA details", "lease terms")
        document_type: Optional filter by document type (OM, DD, CA)

    Returns:
        List of relevant document chunks with their URLs and content
    """
    try:
        # Generate embeddings for the query
        query_embeddings = await ai_flows.generate_embeddings(query)

        # Build search filter for the specific property
        search_filter = f"property_id eq '{property_id}'"
        if document_type:
            search_filter += f" and document_type eq '{document_type}'"

        # Search for relevant documents using vector search + text search
        results = await azure_manager.search_documents(
            query=query,
            vector=query_embeddings,
            index_name=settings.azure_property_document_index,
            filter_expression=search_filter,
            top=10,
        )

        document_chunks = []
        for result in results:
            document_chunks.append(
                {
                    "chunk_text": result.get("chunk_text", ""),
                    "document_url": result.get("document_url", ""),
                    "document_type": result.get("document_type", ""),
                    "document_name": result.get("document_name", ""),
                    "property_name": result.get("property_name", ""),
                    "chunk_index": result.get("chunk_index", 0),
                    "total_chunks": result.get("total_chunks", 1),
                    "score": result.get("@search.score", 0.0),
                }
            )

        logger.info(
            f"Retrieved {len(document_chunks)} document chunks for property {property_id}"
        )
        return document_chunks

    except Exception as e:
        logger.error(f"Error retrieving property documents for {property_id}: {str(e)}")
        return []


async def search_documents_by_property_name(
    property_name: str, query: str, document_type: str = None
) -> List[Dict]:
    """
    Search for documents by property name (for natural language queries)
    Enables queries like: "what are the NDA details of Ocean Park Plaza?"

    Args:
        property_name: Name of the property to search
        query: The search query content
        document_type: Optional document type filter (OM, DD, CA)

    Returns:
        List of relevant document chunks
    """
    try:
        # Generate embeddings for the query
        query_embeddings = await ai_flows.generate_embeddings(query)

        # Build search filter for property name
        search_filter = f"search.ismatch('{property_name}', 'property_name')"
        if document_type:
            search_filter += f" and document_type eq '{document_type}'"

        # Search for relevant documents
        results = await azure_manager.search_documents(
            query=query,
            vector=query_embeddings,
            index_name=settings.azure_property_document_index,
            filter_expression=search_filter,
            top=10,
        )

        document_chunks = []
        for result in results:
            document_chunks.append(
                {
                    "chunk_text": result.get("chunk_text", ""),
                    "document_url": result.get("document_url", ""),
                    "document_type": result.get("document_type", ""),
                    "document_name": result.get("document_name", ""),
                    "property_name": result.get("property_name", ""),
                    "property_id": result.get("property_id", ""),
                    "chunk_index": result.get("chunk_index", 0),
                    "total_chunks": result.get("total_chunks", 1),
                    "score": result.get("@search.score", 0.0),
                }
            )

        logger.info(
            f"Retrieved {len(document_chunks)} document chunks for property '{property_name}'"
        )
        return document_chunks

    except Exception as e:
        logger.error(
            f"Error searching documents for property '{property_name}': {str(e)}"
        )
        return []
