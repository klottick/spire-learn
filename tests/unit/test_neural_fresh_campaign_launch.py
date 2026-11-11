"""Current authored qualification selection without any runtime effects."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import replace

from scripts.neural import curriculum_controls as controls
from scripts.neural import curriculum_profiles as profiles


class FreshCampaignLaunchTests(unittest.TestCase):
    def test_current_runtime_installs_fault_capture_before_return(self):
        from scripts.neural import curriculum_runtime as producer, curriculum_fault_capture as capture
        from spire_learn.neural import runner
        from spire_learn.neural.config import load_neural_config
        config=load_neural_config(Path('configs/neural/experiments/curriculum-v4/bootstrap.toml'))
        calls=[]
        def install(actual,output):calls.append((actual,output));return output/'capture'
        with tempfile.TemporaryDirectory() as temporary:
            config=replace(config,output_root=Path(temporary))
            runtime=SimpleNamespace(config=config,close=lambda:None)
            with profiles.scope('fresh-gym-afb51f5f'),patch.object(runner,'create_runtime',return_value=runtime), \
                 patch.object(producer,'_threads',return_value={'intraop':4,'interop':1}), \
                 patch.object(capture,'install_fault_capture',side_effect=install):
                result=producer._factory(config,purpose='train')
            self.assertEqual(calls,[(runtime,Path(temporary))])
            self.assertEqual(result.fault_capture_directory,Path(temporary)/'capture')

    def test_qualification_inventory_uses_current_authored_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); actual=root/'actual';authored=root/'authored'
            actual.mkdir();authored.mkdir()
            for directory in (actual,authored):
                (directory/'qualification-0.toml').write_text('current = true\n')
            selected=SimpleNamespace(qualification_input=Path('actual'),fixed=True)
            with patch.object(profiles,'current',return_value=selected),patch.object(
                    profiles,'qualification_source',return_value=authored):
                result=controls.qualification_inventory(root)
            self.assertEqual(set(result),{'qualification-0.toml'})
            self.assertEqual(result['qualification-0.toml']['size_bytes'],15)

    def test_qualification_api_proxy_captures_runtime_without_mutating_held_namespace(self):
        from scripts.neural import curriculum_runtime as producer, curriculum_fault_capture as capture
        events=[]
        with tempfile.TemporaryDirectory() as temporary:
            config=SimpleNamespace(output_root=Path(temporary))
            runtime=SimpleNamespace(config=config, collector=SimpleNamespace(driver=object()),
                                    close=lambda:None)
            def create_runtime(*args,**kwargs):
                events.append(('factory',args,kwargs))
                return runtime
            original=SimpleNamespace(create_runtime=create_runtime, marker=object())
            def install(actual,output):
                events.append(('capture',actual,output))
                return output/'capture'
            with profiles.scope('fresh-gym-afb51f5f'),patch.object(
                    capture,'install_fault_capture',side_effect=install):
                wrapped=producer._wrap_qualification_api(original)
                result=wrapped.create_runtime(config,purpose='resume')
            self.assertIsNot(wrapped,original)
            self.assertIs(original.create_runtime,create_runtime)
            self.assertIs(wrapped.marker,original.marker)
            self.assertIs(result,runtime)
            self.assertEqual(events,[('factory',(config,),{'purpose':'resume'}),
                ('capture',runtime,Path(temporary))])
            self.assertEqual(result.fault_capture_directory,Path(temporary)/'capture')

    def test_qualification_capture_failure_closes_runtime_once_and_preserves_primary_controls(self):
        from scripts.neural import curriculum_runtime as producer, curriculum_fault_capture as capture
        primary=RuntimeError('fault capture failed')
        close_count=[]
        with tempfile.TemporaryDirectory() as temporary:
            config=SimpleNamespace(output_root=Path(temporary))
            def close(): close_count.append(True)
            runtime=SimpleNamespace(config=config,collector=SimpleNamespace(driver=object()),close=close)
            original=SimpleNamespace(create_runtime=lambda *args,**kwargs:runtime)
            with profiles.scope('fresh-gym-afb51f5f'),patch.object(
                    capture,'install_fault_capture',side_effect=primary):
                wrapped=producer._wrap_qualification_api(original)
                with self.assertRaises(RuntimeError) as raised:
                    wrapped.create_runtime(config,purpose='resume')
        self.assertIs(raised.exception,primary)
        self.assertEqual(close_count,[True])

        close_count=[]
        primary_control=KeyboardInterrupt('primary capture control')
        secondary_control=SystemExit('close control')
        with tempfile.TemporaryDirectory() as temporary:
            config=SimpleNamespace(output_root=Path(temporary))
            def close_with_control():
                close_count.append(True)
                raise secondary_control
            runtime=SimpleNamespace(config=config,collector=SimpleNamespace(driver=object()),
                                    close=close_with_control)
            original=SimpleNamespace(create_runtime=lambda *args,**kwargs:runtime)
            with profiles.scope('fresh-gym-afb51f5f'),patch.object(
                    capture,'install_fault_capture',side_effect=primary_control):
                wrapped=producer._wrap_qualification_api(original)
                with self.assertRaises(KeyboardInterrupt) as raised:
                    wrapped.create_runtime(config,purpose='resume')
        self.assertIs(raised.exception,primary_control)
        self.assertEqual(close_count,[True])
        self.assertTrue(any('SystemExit' in note for note in primary_control.__notes__))

    def test_legacy_qualification_api_is_returned_unchanged(self):
        from scripts.neural import curriculum_runtime as producer
        original=SimpleNamespace(create_runtime=lambda:object(), marker=object())
        with profiles.scope('historical'):
            wrapped=producer._wrap_qualification_api(original)
        self.assertIs(wrapped,original)

    def test_qualification_api_proxy_reuses_existing_capture(self):
        from scripts.neural import curriculum_runtime as producer, curriculum_fault_capture as capture
        with tempfile.TemporaryDirectory() as temporary:
            existing=Path(temporary)/'already-captured'
            runtime=SimpleNamespace(config=SimpleNamespace(output_root=Path(temporary)),
                collector=SimpleNamespace(driver=object()),fault_capture_directory=existing,
                close=lambda:None)
            original=SimpleNamespace(create_runtime=lambda *args,**kwargs:runtime)
            with profiles.scope('fresh-gym-afb51f5f'),patch.object(
                    capture,'install_fault_capture') as install:
                wrapped=producer._wrap_qualification_api(original)
                result=wrapped.create_runtime(runtime.config,purpose='resume')
            install.assert_not_called()
            self.assertIs(result.fault_capture_directory,existing)
