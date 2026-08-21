import json
import unittest

from Utils.langfuse_tracing import BadmintonTracer


class LocalPromptTracingTest(unittest.TestCase):
    def test_capture_records_execution_without_system_prompt(self):
        tracer = BadmintonTracer(
            'naive_rag',
            'A',
            enable_tracing=False,
            capture_local=True,
        )
        tracer.start_trace(
            {'opponent_type': 'Serve short', 'player_location_area': 5},
            {
                'match_id': 7,
                'set': 1,
                'rally': 2,
                'round': [3],
                'score': [4, 5],
            },
            launch=False,
        )
        query = {'opponent_type': 'Serve short'}
        cases = [{'case_id': 9, 'distance': 0.25}]
        tracer.log_retrieval(query, cases, 'generic', latency_ms=3, stage='stage1')
        response = {'outcome': 'returned', 'shot_type': 'Lob'}
        tracer.log_generation(
            'SYSTEM PROMPT MUST NOT BE SAVED',
            'Rendered user prompt',
            response,
            latency_ms=11,
            raw_llm_output='{"outcome":"returned","shot_type":"Lob"}',
            usage={'total_tokens': 17},
            stage='stage1',
        )
        tracer.end_trace({'shot_type': 8}, 'Returned with a lob.')
        trace = tracer.consume_local_trace()

        query['opponent_type'] = 'mutated'
        cases[0]['case_id'] = 10
        response['shot_type'] = 'Clear'
        payload = json.dumps(trace)
        self.assertNotIn('SYSTEM PROMPT MUST NOT BE SAVED', payload)
        self.assertNotIn('system_prompt', payload)
        self.assertEqual(trace['retrievals'][0]['query']['opponent_type'], 'Serve short')
        self.assertEqual(trace['retrievals'][0]['cases'][0]['case_id'], 9)
        self.assertEqual(trace['generations'][0]['parsed_response']['shot_type'], 'Lob')
        self.assertEqual(trace['generations'][0]['usage']['total_tokens'], 17)
        self.assertIsNone(tracer.consume_local_trace())


if __name__ == '__main__':
    unittest.main()
