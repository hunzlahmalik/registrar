"""
Utility objects for communicating with other web services
"""
import logging
from posixpath import join as urljoin

from django.conf import settings
from requests.exceptions import HTTPError

from .rest_utils import get_all_paginated_results, make_request


logger = logging.getLogger(__name__)
DISCOVERY_API_TPL = 'api/v1/{}/{}'


class DiscoveryServiceClient:
    """ The Discovery service API client to make the RESTFul API call """
    @classmethod
    def get_program(cls, uuid):
        """
        Fetch a JSON representation of a program from the Discovery service.

        Returns None if not found or other HTTP error.

        Arguments:
            * uuid (UUID)

        Returns: dict|None
        """
        url = urljoin(
            settings.DISCOVERY_BASE_URL,
            DISCOVERY_API_TPL.format('programs', uuid)
        )
        try:
            return make_request('GET', url, client=None).json()
        except HTTPError:
            logger.exception(
                "Failed to load program with uuid %s from Discovery service.",
                uuid,
            )
            return None

    @classmethod
    def get_programs_by_types(cls, types):
        """
        Fetch a JSON representation of all active programs of specified
        types from the Discovery service.

        Returns empty if not found or other HTTP error.

        Arguments:
            * types: [program_types]

        Returns: [programs] | []
        """
        url = urljoin(
            settings.DISCOVERY_BASE_URL,
            DISCOVERY_API_TPL.format('programs', '')
        )
        url += f'?types={",".join(types)}&status=active'
        try:
            return get_all_paginated_results(url)
        except HTTPError:
            logger.exception(
                "Failed to load programs with program_types %s from "
                "Discovery service.",
                ','.join(types),
            )
            return []

    @classmethod
    def get_programs_by_uuids(cls, uuids):
        """
        Fetch a JSON representation of programs by their UUIDs from the
        Discovery service.

        Returns empty if not found or other HTTP error.

        Arguments:
            * uuids: [program_uuids]

        Returns: [programs] | []
        """
        if not uuids:
            return []

        url = urljoin(
            settings.DISCOVERY_BASE_URL,
            DISCOVERY_API_TPL.format('programs', '')
        )
        url += f'?uuids={",".join(str(uuid) for uuid in uuids)}'
        try:
            return get_all_paginated_results(url)
        except HTTPError:
            logger.exception(
                "Failed to load programs with uuids %s from "
                "Discovery service.",
                ','.join(str(uuid) for uuid in uuids),
            )
            return []

    @classmethod
    def get_organizations(cls):
        """
        Fetch a JSON representation of organizations from the Discovery
        service.

        Returns None if not found or other HTTP error.

        Arguments:
            * None

        Returns: [organization] | []
        """
        url = urljoin(
            settings.DISCOVERY_BASE_URL,
            DISCOVERY_API_TPL.format('organizations', '')
        )
        try:
            return get_all_paginated_results(url)
        except HTTPError:
            logger.exception(
                "Failed to load organizations from Discovery service.",
            )
            return []
