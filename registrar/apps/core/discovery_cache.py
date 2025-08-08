"""
Simple interface to program details from Course Discovery data through a volatile cache.
"""
import logging
from collections import namedtuple
from uuid import UUID

from django.conf import settings
from django.core.cache import cache

from .api_client import DiscoveryServiceClient
from .constants import PROGRAM_CACHE_KEY_TPL


logger = logging.getLogger(__name__)

DiscoveryCourseRun = namedtuple(
    'DiscoveryCourseRun',
    ['key', 'external_key', 'title', 'marketing_url'],
)


class ProgramDetails:
    """
    Details about a program from the Course Discovery service.

    Data from Discovery is cached for `settings.PROGRAM_CACHE_TIMEOUT` seconds.

    If Discovery data cannot be loaded, we quietly fall back to an object that returns
    default values (generally empty dicts and None) instead of raising an exception.
    Callers should anticipate that Discovery data may not always be available,
    and use the default values gracefully.

    Details are loaded in the call to the constructor.

    Example usage:
        details = ProgramDetails(program.discovery_uuid)
        if details.find_course_run(course_key):
            # ...
    """

    @classmethod
    def _get_default_chunk_size(cls):
        """
        Get the default chunk size from environment settings.

        Returns:
            int: The chunk size from REGISTRAR_DISCOVERY_PROGRAMS_API_BATCH_SIZE setting,
                 or 20 if not configured.
        """
        return getattr(settings, 'REGISTRAR_DISCOVERY_PROGRAMS_API_BATCH_SIZE', 20)

    def __init__(self, uuid):
        """
        Initialize a program details instance and load details from cache or Discovery.

        Arguments:
            uuid (UUID|str): UUID of the program as defined in Discovery service.
        """
        self.uuid = uuid
        self.raw_data = self.get_raw_data_for_program(uuid)

    @classmethod
    def get_raw_data_for_program(cls, uuid):
        """
        Retrieve JSON data containing program details, looking in cache first.

        This is the access point to the "Discovery cache" as referenced
        in comments throughout Registrar.

        Note that "not-found-in-discovery" is purposefully cached as `{}`,
        whereas `cache.get(key)` will return `None` if the key is not in the
        cache.

        Arguments:
            uuid (UUID|str): UUID of the program as defined in Discovery service.

        Returns: dict
        """
        cache_key = PROGRAM_CACHE_KEY_TPL.format(uuid=uuid)
        data = cache.get(cache_key)
        if not isinstance(data, dict):
            data = DiscoveryServiceClient.get_program(uuid)
            if not isinstance(data, dict):
                data = {}
            cache.set(cache_key, data, settings.PROGRAM_CACHE_TIMEOUT)
        return data

    @classmethod
    def _get_raw_data_for_programs(cls, uuids, chunk_size=None):
        """
        Retrieve JSON data for multiple programs, using bulk fetching and caching.

        This method efficiently fetches multiple programs by:
        1. Checking cache for existing programs
        2. Bulk fetching missing programs from Discovery API
        3. Caching all fetched results

        Arguments:
            uuids (Iterable[UUID|str]): UUIDs of programs to fetch
            chunk_size (int): Maximum UUIDs per API request
                (default: from settings.REGISTRAR_DISCOVERY_PROGRAMS_API_BATCH_SIZE)

        Returns: dict[str, dict] - Mapping of UUID strings to program data
        """
        if not uuids:
            return {}

        # Convert all UUIDs to strings for consistency
        str_uuids = [str(uuid) for uuid in uuids]
        programs_dict = {}
        uuids_to_fetch = []

        # Check cache for existing programs
        cache_keys = [PROGRAM_CACHE_KEY_TPL.format(uuid=uuid) for uuid in str_uuids]
        cached_programs = cache.get_many(cache_keys)

        for uuid_str in str_uuids:
            cache_key = PROGRAM_CACHE_KEY_TPL.format(uuid=uuid_str)
            if cache_key in cached_programs and isinstance(cached_programs[cache_key], dict):
                programs_dict[uuid_str] = cached_programs[cache_key]
            else:
                uuids_to_fetch.append(uuid_str)

        # Fetch missing programs from API in chunks
        if uuids_to_fetch:
            fetched_programs = cls._fetch_programs_bulk(uuids_to_fetch, chunk_size)

            # Process fetched programs and prepare for caching
            cache_data = {}
            found_uuids = set()

            for program in fetched_programs:
                program_uuid = str(program.get('uuid'))
                if program_uuid:
                    programs_dict[program_uuid] = program
                    cache_key = PROGRAM_CACHE_KEY_TPL.format(uuid=program_uuid)
                    cache_data[cache_key] = program
                    found_uuids.add(program_uuid)

            # Cache fetched programs
            if cache_data:
                cache.set_many(cache_data, settings.PROGRAM_CACHE_TIMEOUT)

            # Missing UUIDs that weren't found in Discovery Programs
            missing_uuids = set(uuids_to_fetch) - found_uuids
            if missing_uuids:
                # Add empty dicts to result for missing programs
                for uuid in missing_uuids:
                    programs_dict[uuid] = {}

        return programs_dict

    @classmethod
    def _fetch_programs_bulk(cls, uuids, chunk_size=None):
        """
        Fetch raw program data for multiple UUIDs from Discovery API in batches.

        Arguments:
            uuids (Iterable[UUID|str]): UUIDs of programs to fetch
            chunk_size (int, optional): Maximum UUIDs per API request.
                                      If None, uses environment setting.

        Returns: list[dict] - List of program data dictionaries
        """
        if chunk_size is None:
            chunk_size = cls._get_default_chunk_size()
        all_programs = []
        # Process UUIDs in chunks to stay within URL length limits
        for i in range(0, len(uuids), chunk_size):
            chunk_uuids = uuids[i:i + chunk_size]

            try:
                chunk_programs = DiscoveryServiceClient.get_programs_by_uuids(chunk_uuids)
                if isinstance(chunk_programs, list):
                    all_programs.extend(chunk_programs)
            except Exception:
                logger.exception(
                    "Failed to load programs chunk with uuids %s from Discovery service.",
                    ','.join(chunk_uuids),
                )
                continue

        return all_programs

    @classmethod
    def clear_cache_for_programs(cls, program_uuids):
        """
        Clear any details from Discovery that we have cached for the given programs.

        Arguments:
            program_uuids (Iterable[str|UUID])
        """
        cache_keys_to_delete = [
            PROGRAM_CACHE_KEY_TPL.format(uuid=program_uuid)
            for program_uuid in program_uuids
        ]
        cache.delete_many(cache_keys_to_delete)

    @classmethod
    def load_many(cls, uuids, chunk_size=None):
        """
        Load multiple ProgramDetails objects for the given program UUIDs.

        Arguments:
            uuids (Iterable[UUID|str]): UUIDs of programs to load
            chunk_size (int): Maximum UUIDs per API request
                (default: from settings.REGISTRAR_DISCOVERY_PROGRAMS_API_BATCH_SIZE)

        Returns: dict[UUID, ProgramDetails] - Mapping of UUIDs to ProgramDetails objects
        """
        if not uuids:
            return {}

        # Convert UUIDs to consistent string format for processing
        uuid_list = [str(uuid) for uuid in uuids]

        # Get chunk size from environment if not provided
        if chunk_size is None:
            chunk_size = cls._get_default_chunk_size()

        # Bulk fetch raw data for all programs
        programs_raw_data = cls._get_raw_data_for_programs(uuid_list, chunk_size)

        # Create ProgramDetails instances using the bulk-fetched data
        result = {}
        for original_uuid in uuids:
            str_uuid = str(original_uuid)
            # Create instance with pre-fetched data to avoid individual API calls
            program_details = cls._create_with_raw_data(original_uuid, programs_raw_data.get(str_uuid, {}))
            result[original_uuid] = program_details

        return result

    @classmethod
    def _create_with_raw_data(cls, uuid, raw_data):
        """
        Create a ProgramDetails instance with pre-fetched raw data.

        This bypasses the normal constructor's API call since we already have the data.

        Arguments:
            uuid (UUID|str): UUID of the program
            raw_data (dict): Pre-fetched program data from Discovery API

        Returns: ProgramDetails
        """
        instance = cls.__new__(cls)  # Create instance without calling __init__
        instance.uuid = uuid
        instance.raw_data = raw_data if isinstance(raw_data, dict) else {}
        return instance

    @property
    def title(self):
        """
        Return title of program.

        Falls back to None if unavailable.
        """
        return self.raw_data.get('title')

    @property
    def url(self):
        """
        Return URL of program

        Falls back to None if unavailable.
        """
        return self.raw_data.get('marketing_url')

    @property
    def program_type(self):
        """
        Return type of the program ("Masters", "MicroMasters", etc.).

        Falls back to None if unavailable.
        """
        return self.raw_data.get('type')

    @property
    def is_enrollment_enabled(self):
        """
        Return whether enrollment is enabled for this program.

        Falls back to False if required data unavailable.

        Currently, enrollment is enabled if and only if the program is a Master's
        degree. This may change in the future.
        """
        return self.program_type == 'Masters'

    @property
    def active_curriculum_details(self):
        """
        Return dict containing details for active curriculum.

        TODO:
        We define 'active' curriculum as the first one in the list of
        curricula where is_active is True.
        This is a temporary assumption, originally made in March 2019.
        We expect that future programs may have more than one curriculum
        active simultaneously, which will require modifying the API.

        Falls back to empty dict if no active curriculum or if data unavailable.
        """
        try:
            return next(
                c for c in self.raw_data.get('curricula', [])
                if c.get('is_active')
            )
        except StopIteration:
            logger.exception(
                'Discovery API returned no active curricula for program %s',
                self.uuid,
            )
            return {}

    @property
    def active_curriculum_uuid(self):
        """
        Get UUID string of active curriculum, or None if no active curriculum.

        See `active_curriculum_details` docstring for more details.
        """
        try:
            return UUID(self.active_curriculum_details.get('uuid'))
        except (TypeError, ValueError):
            return None

    @property
    def course_runs(self):
        """
        Get list of DiscoveryCourseRuns defined in root of active curriculum.

        TODO:
        In March 2019 we made a temporary assumption that the curriculum
        does not contain nested programs.
        We expect that this will need revisiting eventually,
        as future programs may have more than one curriculum.

        Also see `active_curriculum_details` docstring details on how the 'active'
        curriculum is determined.

        Falls back to empty list if no active curriculum or if data unavailable.
        """
        return [
            DiscoveryCourseRun(
                key=course_run.get("key"),
                external_key=course_run.get("external_key"),
                title=course_run.get("title"),
                marketing_url=course_run.get("marketing_url"),
            )
            for course in self.active_curriculum_details.get("courses", [])
            for course_run in course.get("course_runs", [])
        ]

    def find_course_run(self, course_id):
        """
        Given a course id, return the course_run with that `key` or `external_key`

        Returns None if course run is not found in the cached program.
        """
        try:
            return next(
                course_run for course_run in self.course_runs
                if course_id in {course_run.key, course_run.external_key}
            )
        except StopIteration:
            return None

    def get_external_course_key(self, course_id):
        """
        Given a course ID, return the external course key for that course_run.
        The course key passed in may be an external or internal course key.

        Returns None if course run is not found in the cached program.
        """
        course_run = self.find_course_run(course_id)
        if course_run:
            return course_run.external_key
        return None

    def get_course_key(self, course_id):
        """
        Given a course ID, return the internal course ID for that course run.
        The course ID passed in may be an external or internal course key.

        Returns None if course run is not found in the cached program.
        """
        course_run = self.find_course_run(course_id)
        if course_run:
            return course_run.key
        return None
