"""Guard the sandbox authority boundary; final signed-app UAT remains required."""
from pathlib import Path


def test_every_model_export_requests_parent_directory_authority():
    source = (Path(__file__).resolve().parents[2] / 'Sources/KwipMLApp/WorkflowWorkspaceSession.swift').read_text()
    selection = source.split('func selectSaveModelDestination(', 1)[1].split('func clearSaveModelDestination(', 1)[0]
    # A save-panel leaf extension cannot authorize sibling staging/atomic publication.
    assert 'NSSavePanel' not in selection, 'All model formats need an explicit parent-directory grant'
    assert 'WorkflowExportParentChooser.panel' in selection
    assert 'validateExclusiveLeaf' in selection
    assert 'modelDestinations.set' in selection


def test_package_routes_share_parent_chooser_but_direct_source_file_keeps_leaf_save():
    root = Path(__file__).resolve().parents[2] / 'Sources/KwipMLApp'
    for name in ['WorkflowTrainedExportView.swift', 'WorkflowProgramExportView.swift',
                 'WorkflowEvaluationExportView.swift', 'CodeActions.swift']:
        source = (root / name).read_text()
        assert 'WorkflowExportParentChooser.panel' in source, name
        assert 'NSOpenPanel()' not in source, name
    code = (root / 'CodeActions.swift').read_text()
    single_file = code.split('func savePythonCode(', 1)[1].split('func exportWorkflowCodePackage()', 1)[0]
    assert 'NSSavePanel' in single_file
    assert 'startAccessingSecurityScopedResource()' in code
    assert 'defer { if accessingParent { parent.stopAccessingSecurityScopedResource() } }' in code
