"""
Tests for core proxy models.
"""

from posixpath import join as urljoin
from unittest.mock import patch
from uuid import UUID, uuid4

import ddt
import responses
from django.conf import settings
from django.core.cache import cache
from django.test import TestCase, override_settings

from ..api_client import DISCOVERY_API_TPL, DiscoveryServiceClient
from ..discovery_cache import ProgramDetails
from .utils import mock_oauth_login


def make_course_run(n, with_external_key=False):
    """
    Return a course run dictionary for testing.

    If `with_external_key` is set to True, set ext key to testorg-course-${n}.
    """
    return {
        'key': f'course-v1:TestRun+{n}',
        'external_key': (
            f'testorg-course-{n}'
            if with_external_key
            else None
        ),
        'title': f'Test Course {n}',
        'marketing_url': f'https://stem.edx.org/masters-in-cs/test-course-{n}',
        'extraneous_data': ['blah blah blah'],
    }


def patch_discovery_client_get_program(mock_response_data):
    """
    Patch the function that we use to call the Discovery service
    to instead statically return `mock_response_data`.

    Note that the responses will still be stored in the Django cache.
    """
    return patch.object(
        DiscoveryServiceClient,
        'get_program',
        lambda *_args, **_kwargs: mock_response_data,
    )


@ddt.ddt
class ProgramDetailsTestCase(TestCase):
    """
    Test ProgramDetails interface to the Discovery data cache.
    """
    program_uuid = UUID("88888888-4444-2222-1111-000000000000")
    discovery_url = urljoin(
        settings.DISCOVERY_BASE_URL,
        DISCOVERY_API_TPL.format('programs', program_uuid)
    )

    inactive_curriculum_uuid = UUID("77777777-4444-2222-1111-000000000000")
    active_curriculum_uuid = UUID(
        "77777777-4444-2222-1111-000000000001"
    )
    ignored_active_curriculum_uuid = UUID(
        "77777777-4444-2222-1111-000000000002"
    )

    program_from_discovery = {
        'title': "Master's in CS",
        'marketing_url': "https://stem.edx.org/masters-in-cs",
        'type': "Masters",
        'curricula': [
            {
                # Inactive curriculum. Should be ignored.
                'uuid': str(inactive_curriculum_uuid),
                'is_active': False,
                'courses': [
                    {'course_runs': [make_course_run(0, True)]}
                ],
            },
            {
                # Active curriculum. All three course runs should be
                # aggregated.
                'uuid': str(active_curriculum_uuid),
                'is_active': True,
                'courses': [
                    {'course_runs': [make_course_run(1)]},
                    {'course_runs': []},
                    {
                        'course_runs': [
                            make_course_run(2, True),
                            make_course_run(3)
                        ]
                    },
                    {
                        'course_runs': [
                            {'this course run': 'has the wrong format'}
                        ]
                    },
                ]
            },
            {
                # Second active curriculum.
                # In current implementation, should be ignored.
                'uuid': str(ignored_active_curriculum_uuid),
                'is_active': True,
                'courses': [
                    {'course_runs': [make_course_run(4, True)]}
                ],
            },
        ],
    }

    def setUp(self):
        super().setUp()
        cache.clear()

    @mock_oauth_login
    @responses.activate
    @ddt.data(
        (200, program_from_discovery, program_from_discovery),
        (200, {}, {}),
        (200, 'this is a string, but it should be a dict', {}),
        (404, {'message': 'program not found'}, {}),
        (500, {'message': 'everything is broken'}, {}),
    )
    @ddt.unpack
    def test_discovery_program_get(
        self, disco_status, disco_json, expected_raw_data
    ):
        responses.add(
            responses.GET,
            self.discovery_url,
            status=disco_status,
            json=disco_json,
        )
        loaded_program = ProgramDetails(self.program_uuid)
        assert isinstance(loaded_program, ProgramDetails)
        assert loaded_program.uuid == self.program_uuid
        assert loaded_program.raw_data == expected_raw_data
        self.assertEqual(len(responses.calls), 1)

        # This should used the cached Discovery response.
        reloaded_program = ProgramDetails(self.program_uuid)
        assert isinstance(reloaded_program, ProgramDetails)
        assert reloaded_program.uuid == self.program_uuid
        assert reloaded_program.raw_data == expected_raw_data
        self.assertEqual(len(responses.calls), 1)

    @patch_discovery_client_get_program(program_from_discovery)
    def test_active_curriculum(self):
        program = ProgramDetails(self.program_uuid)
        assert program.active_curriculum_uuid == self.active_curriculum_uuid
        assert len(program.course_runs) == 4
        assert program.course_runs[0].title == "Test Course 1"
        assert program.course_runs[-1].title is None

    @patch_discovery_client_get_program({})
    def test_no_active_curriculum(self):
        program = ProgramDetails(self.program_uuid)
        assert program.active_curriculum_uuid is None
        assert not program.course_runs

    @patch_discovery_client_get_program(program_from_discovery)
    @ddt.data(
        # Non-existent course run.
        ('non-existent', None, None),
        # Real course run, but not in active curriculum.
        ('course-v1:TestRun+0', None, None),
        ('testorg-course-0', None, None),
        # Real course run, in active curriculum, only has internal course key.
        ('course-v1:TestRun+1', 'course-v1:TestRun+1', None),
        ('testorg-course-1', None, None),
        # Real course run, in active curriculum, has internal and
        # external keys.
        (
            'course-v1:TestRun+2',
            'course-v1:TestRun+2',
            'testorg-course-2'
        ),
        (
            'testorg-course-2',
            'course-v1:TestRun+2',
            'testorg-course-2'
        ),
    )
    @ddt.unpack
    def test_get_course_keys(
        self, argument, expected_course_key, expected_external_key
    ):
        program = ProgramDetails(self.program_uuid)
        actual_course_key = program.get_course_key(argument)
        assert actual_course_key == expected_course_key
        actual_external_key = program.get_external_course_key(argument)
        assert actual_external_key == expected_external_key


class ProgramDetailsBulkFetchingTestCase(TestCase):
    """
    Test ProgramDetails bulk fetching functionality.
    """
    # pylint: disable=protected-access  # Testing private methods
    program_uuid_1 = UUID("11111111-1111-1111-1111-111111111111")
    program_uuid_2 = UUID("22222222-2222-2222-2222-222222222222")
    program_uuid_3 = UUID("33333333-3333-3333-3333-333333333333")

    program_1_data = {
        'uuid': str(program_uuid_1),
        'title': "Test Program 1",
        'type': "Masters",
        'marketing_url': "https://example.com/program1",
    }

    program_2_data = {
        'uuid': str(program_uuid_2),
        'title': "Test Program 2",
        'type': "MicroMasters",
        'marketing_url': "https://example.com/program2",
    }

    program_3_data = {
        'uuid': str(program_uuid_3),
        'title': "Test Program 3",
        'type': "Masters",
        'marketing_url': "https://example.com/program3",
    }

    def setUp(self):
        super().setUp()
        cache.clear()

    def test_get_raw_data_for_programs_empty_list(self):
        """Test that empty list returns empty dict."""
        result = ProgramDetails._get_raw_data_for_programs([])
        self.assertEqual(result, {})

    @patch.object(DiscoveryServiceClient, 'get_programs_by_uuids')
    def test_get_raw_data_for_programs_all_cached(self, mock_get_programs):
        """Test that cached programs don't trigger API calls."""
        # Pre-populate cache
        cache.set(
            f'program:{self.program_uuid_1}', self.program_1_data, 300
        )
        cache.set(
            f'program:{self.program_uuid_2}', self.program_2_data, 300
        )

        uuids = [self.program_uuid_1, self.program_uuid_2]
        result = ProgramDetails._get_raw_data_for_programs(uuids)

        # Should not call API
        mock_get_programs.assert_not_called()

        # Should return cached data
        expected = {
            str(self.program_uuid_1): self.program_1_data,
            str(self.program_uuid_2): self.program_2_data,
        }
        self.assertEqual(result, expected)

    @patch.object(DiscoveryServiceClient, 'get_programs_by_uuids')
    def test_get_raw_data_for_programs_none_cached(self, mock_get_programs):
        """Test that uncached programs trigger API calls and get cached."""
        mock_get_programs.return_value = [
            self.program_1_data, self.program_2_data
        ]

        uuids = [self.program_uuid_1, self.program_uuid_2]
        result = ProgramDetails._get_raw_data_for_programs(uuids)

        # Should call API once
        mock_get_programs.assert_called_once_with([
            str(self.program_uuid_1), str(self.program_uuid_2)
        ])

        # Should return API data
        expected = {
            str(self.program_uuid_1): self.program_1_data,
            str(self.program_uuid_2): self.program_2_data,
        }
        self.assertEqual(result, expected)

        # Data should be cached
        self.assertEqual(
            cache.get(f'program:{self.program_uuid_1}'), self.program_1_data
        )
        self.assertEqual(
            cache.get(f'program:{self.program_uuid_2}'), self.program_2_data
        )

    @patch.object(DiscoveryServiceClient, 'get_programs_by_uuids')
    def test_get_raw_data_for_programs_mixed_cache(self, mock_get_programs):
        """Test that mix of cached and uncached programs works correctly."""
        # Pre-populate cache with one program
        cache.set(
            f'program:{self.program_uuid_1}', self.program_1_data, 300
        )

        # Mock API to return only the uncached program
        mock_get_programs.return_value = [self.program_2_data]

        uuids = [self.program_uuid_1, self.program_uuid_2]
        result = ProgramDetails._get_raw_data_for_programs(uuids)

        # Should call API only for uncached program
        mock_get_programs.assert_called_once_with([
            str(self.program_uuid_2)
        ])

        # Should return both cached and API data
        expected = {
            str(self.program_uuid_1): self.program_1_data,
            str(self.program_uuid_2): self.program_2_data,
        }
        self.assertEqual(result, expected)

    @patch.object(DiscoveryServiceClient, 'get_programs_by_uuids')
    def test_get_raw_data_for_programs_chunking(self, mock_get_programs):
        """Test that large UUID lists are chunked properly."""
        # Create 25 UUIDs (more than default chunk size of 20)
        uuids = [
            UUID(f"{i:08d}-1111-1111-1111-111111111111")
            for i in range(25)
        ]

        # Mock API to return empty list for simplicity
        mock_get_programs.return_value = []
        ProgramDetails._get_raw_data_for_programs(uuids, chunk_size=10)

        # Should be called 3 times (25 UUIDs / 10 per chunk = 3 chunks)
        self.assertEqual(mock_get_programs.call_count, 3)

        # Verify chunk sizes
        call_args_list = mock_get_programs.call_args_list
        # First chunk: 10 UUIDs
        self.assertEqual(len(call_args_list[0][0][0]), 10)
        # Second chunk: 10 UUIDs
        self.assertEqual(len(call_args_list[1][0][0]), 10)
        # Third chunk: 5 UUIDs
        self.assertEqual(len(call_args_list[2][0][0]), 5)

    @patch.object(DiscoveryServiceClient, 'get_programs_by_uuids')
    def test_get_raw_data_for_programs_missing_programs(
        self, mock_get_programs
    ):
        """Test handling of programs not found in Discovery."""
        # Mock API to return only one program (missing the second)
        mock_get_programs.return_value = [self.program_1_data]

        uuids = [self.program_uuid_1, self.program_uuid_2]
        result = ProgramDetails._get_raw_data_for_programs(uuids)

        # Should return found program and empty dict for missing
        expected = {
            str(self.program_uuid_1): self.program_1_data,
            str(self.program_uuid_2): {},  # Missing program gets empty dict
        }
        self.assertEqual(result, expected)

        # Found program should be cached
        self.assertEqual(
            cache.get(f'program:{self.program_uuid_1}'), self.program_1_data
        )

        # Missing program should NOT be cached (empty dict not cached)
        self.assertIsNone(
            cache.get(f'program:{self.program_uuid_2}')
        )

    @patch.object(DiscoveryServiceClient, 'get_programs_by_uuids')
    def test_get_raw_data_for_programs_missing_programs_not_cached(
        self, mock_get_programs
    ):
        """Test that missing programs are not cached and trigger API calls on subsequent requests."""
        # Mock API to return empty list (no programs found)
        mock_get_programs.return_value = []

        uuid = self.program_uuid_1

        # First call - should trigger API call
        result1 = ProgramDetails._get_raw_data_for_programs([uuid])
        self.assertEqual(result1, {str(uuid): {}})
        self.assertEqual(mock_get_programs.call_count, 1)

        # Missing program should NOT be cached
        self.assertIsNone(cache.get(f'program:{uuid}'))

        # Second call - should trigger another API call since missing programs aren't cached
        result2 = ProgramDetails._get_raw_data_for_programs([uuid])
        self.assertEqual(result2, {str(uuid): {}})
        self.assertEqual(mock_get_programs.call_count, 2)  # Should be called again

    def test_load_many_empty_list(self):
        """Test that load_many with empty list returns empty dict."""
        result = ProgramDetails.load_many([])
        self.assertEqual(result, {})

    @patch.object(ProgramDetails, '_get_raw_data_for_programs')
    def test_load_many_creates_instances(self, mock_get_raw_data):
        """Test that load_many creates ProgramDetails instances with
        correct data."""
        mock_get_raw_data.return_value = {
            str(self.program_uuid_1): self.program_1_data,
            str(self.program_uuid_2): self.program_2_data,
        }

        uuids = [self.program_uuid_1, self.program_uuid_2]
        result = ProgramDetails.load_many(uuids)

        # Should call bulk fetch method (chunk_size will be from environment)
        self.assertEqual(mock_get_raw_data.call_count, 1)
        call_args = mock_get_raw_data.call_args
        self.assertEqual(call_args[0][0], [str(self.program_uuid_1), str(self.program_uuid_2)])
        # chunk_size is the second argument, verify it's a positive integer
        self.assertIsInstance(call_args[0][1], int)
        self.assertGreater(call_args[0][1], 0)
        # Should return correct number of instances
        self.assertEqual(len(result), 2)
        self.assertIn(self.program_uuid_1, result)
        self.assertIn(self.program_uuid_2, result)

        # Instances should have correct data
        program_1 = result[self.program_uuid_1]
        program_2 = result[self.program_uuid_2]

        self.assertIsInstance(program_1, ProgramDetails)
        self.assertIsInstance(program_2, ProgramDetails)
        self.assertEqual(program_1.uuid, self.program_uuid_1)
        self.assertEqual(program_2.uuid, self.program_uuid_2)
        self.assertEqual(program_1.title, "Test Program 1")
        self.assertEqual(program_2.title, "Test Program 2")

    @patch.object(ProgramDetails, '_get_raw_data_for_programs')
    def test_load_many_passes_chunk_size(self, mock_get_raw_data):
        """Test that load_many passes chunk_size parameter to _get_raw_data_for_programs."""
        mock_get_raw_data.return_value = {
            str(self.program_uuid_1): self.program_1_data,
        }

        uuids = [self.program_uuid_1]
        ProgramDetails.load_many(uuids, chunk_size=15)

        # Should call bulk fetch method with explicit chunk_size
        mock_get_raw_data.assert_called_once_with([
            str(self.program_uuid_1)
        ], 15)

    @patch.object(ProgramDetails, '_get_raw_data_for_programs')
    @override_settings(REGISTRAR_DISCOVERY_PROGRAMS_API_BATCH_SIZE=7)
    def test_load_many_uses_env_batch_size(self, mock_get_raw_data):
        """Test that load_many uses environment batch size when no chunk_size provided."""
        mock_get_raw_data.return_value = {
            str(self.program_uuid_1): self.program_1_data,
        }

        uuids = [self.program_uuid_1]
        ProgramDetails.load_many(uuids)  # No chunk_size parameter

        # Should call bulk fetch method with environment batch size
        mock_get_raw_data.assert_called_once_with([
            str(self.program_uuid_1)
        ], 7)

    def test_create_with_raw_data(self):
        """Test that _create_with_raw_data creates instance correctly."""
        # pylint: disable=protected-access
        program = ProgramDetails._create_with_raw_data(
            self.program_uuid_1, self.program_1_data
        )
        self.assertIsInstance(program, ProgramDetails)
        self.assertEqual(program.uuid, self.program_uuid_1)
        self.assertEqual(program.raw_data, self.program_1_data)
        self.assertEqual(program.title, "Test Program 1")
        self.assertEqual(program.program_type, "Masters")

    def test_create_with_raw_data_empty(self):
        """Test that _create_with_raw_data handles empty data."""
        # pylint: disable=protected-access
        program = ProgramDetails._create_with_raw_data(
            self.program_uuid_1, {}
        )

        self.assertIsInstance(program, ProgramDetails)
        self.assertEqual(program.uuid, self.program_uuid_1)
        self.assertEqual(program.raw_data, {})
        self.assertIsNone(program.title)
        self.assertIsNone(program.program_type)

    def test_clear_cache_for_programs(self):
        """Test that clear_cache_for_programs removes correct cache entries."""
        # Pre-populate cache
        cache.set(
            f'program:{self.program_uuid_1}', self.program_1_data, 300
        )
        cache.set(
            f'program:{self.program_uuid_2}', self.program_2_data, 300
        )
        cache.set(
            f'program:{self.program_uuid_3}', self.program_3_data, 300
        )

        # Clear cache for two programs
        ProgramDetails.clear_cache_for_programs([
            self.program_uuid_1, self.program_uuid_2
        ])

        # Those programs should be cleared
        self.assertIsNone(
            cache.get(f'program:{self.program_uuid_1}')
        )
        self.assertIsNone(
            cache.get(f'program:{self.program_uuid_2}')
        )

        # Third program should remain
        self.assertEqual(
            cache.get(f'program:{self.program_uuid_3}'),
            self.program_3_data
        )

    @patch('registrar.apps.core.discovery_cache.DiscoveryServiceClient.get_programs_by_uuids')
    @override_settings(REGISTRAR_DISCOVERY_PROGRAMS_API_BATCH_SIZE=5)
    def test_get_raw_data_for_programs_uses_env_batch_size(self, mock_get_programs):
        """Test that _get_raw_data_for_programs uses environment batch size setting."""
        # Setup mock to return different programs for different chunks
        def side_effect(uuids):
            return [{'uuid': uuid, 'title': f'Program {uuid}'} for uuid in uuids]
        mock_get_programs.side_effect = side_effect

        # Create 12 UUIDs to force chunking with batch size 5
        uuids = [str(uuid4()) for _ in range(12)]

        # Call with no explicit chunk_size to use environment setting
        ProgramDetails._get_raw_data_for_programs(uuids)

        # Should have made 3 calls (5, 5, 2) based on batch size 5
        self.assertEqual(mock_get_programs.call_count, 3)

        # Check the call arguments
        calls = mock_get_programs.call_args_list
        self.assertEqual(len(calls[0][0][0]), 5)  # First chunk: 5 UUIDs
        self.assertEqual(len(calls[1][0][0]), 5)  # Second chunk: 5 UUIDs
        self.assertEqual(len(calls[2][0][0]), 2)  # Third chunk: 2 UUIDs

    @patch('registrar.apps.core.discovery_cache.DiscoveryServiceClient.get_programs_by_uuids')
    def test_get_raw_data_for_programs_explicit_chunk_size_overrides_env(self, mock_get_programs):
        """Test that explicit chunk_size parameter overrides environment setting."""
        mock_get_programs.return_value = []

        # Create 6 UUIDs
        uuids = [str(uuid4()) for _ in range(6)]

        # Call with explicit chunk_size=3
        ProgramDetails._get_raw_data_for_programs(uuids, chunk_size=3)

        # Should have made 2 calls (3, 3) based on explicit chunk size 3
        self.assertEqual(mock_get_programs.call_count, 2)

        # Check the call arguments
        calls = mock_get_programs.call_args_list
        self.assertEqual(len(calls[0][0][0]), 3)  # First chunk: 3 UUIDs
        self.assertEqual(len(calls[1][0][0]), 3)  # Second chunk: 3 UUIDs
