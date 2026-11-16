import SwiftUI
struct FormulaInformationView:View {
    var body:some View {
        VStack(alignment:.leading,spacing:9) {
            Label("Ordered column operations",systemImage:"list.number").fontWeight(.semibold).foregroundStyle(MLTheme.teal)
            Text("Operations run top to bottom. Later rows may reference earlier outputs using [column]. Drag the grip or use arrows to reorder. Each output name is written once; choose New column or explicitly replace an existing scalar column. Image columns pass through unchanged and cannot be scalar inputs or replacement targets.")
            Label("One expression, two editors",systemImage:"function").fontWeight(.semibold)
            Text("Guided Replace text, Extract fields and Match pattern serialize the same allowlisted regex calls as Expression. Switching to Expression exposes that exact call. A compatible whole regex call can return to Guided. General arithmetic and nested function expressions use Expression. Insert column and Functions act at the expression cursor; they do not run Python.")
            Text("Supported functions: lower, upper, trim, length, abs, round, require, coalesce, regex_replace, regex_extract and regex_match. round uses nearest integer, with ties to even. Number types remain explicit. Formula does not implicitly convert text to numbers or decode image pixels.")
            Text("Use require(condition, value) for a workflow assumption: only Boolean true passes the value through; false or missing conditions stop the run with a diagnostic. Both arguments are evaluated. Use Filter before arithmetic that is unsafe for excluded rows.")
            Label("Exact decimal text",systemImage:"number").fontWeight(.semibold)
            Text("Keep original decimal columns as Text. decimal_parse returns canonical exact text or missing for invalid/nonfinite input. decimal_is_finite and decimal_is_integer check the original text without binary rounding. decimal_add, decimal_subtract, decimal_multiply and decimal_abs return exact text; decimal_compare returns −1, 0 or 1. Invalid arithmetic stops the operation. Missing values follow the chosen policy. Use Summarize → Sum exact decimal text for totals. Ordinary Number arithmetic remains approximate; converting a rounded Number back to Text cannot recover its original digits.")
            Text("Decimal source/result text is limited to 16,384 characters, with bounded exponents. Exceeding an exact-processing allowance fails explicitly; results are never silently rounded or truncated. No automatic currency, scale or cents-rounding policy is applied.")
            Label("Regex and missing values",systemImage:"text.magnifyingglass").fontWeight(.semibold)
            Text("Patterns use the bundled Python regex syntax. Replacement accepts literal text and capture references, including \\1 and \\g<name>. First/All controls replacement count; Contains/Entire controls searching versus full matching. Extraction maps each numbered or named group to a text column. Nonparticipating groups and unmatched rows produce missing values unless their policy stops the run. Missing result applies after the expression, so coalesce can provide an intentional fallback.")
            Text("Flags are off by default. Expressions take flags as a pipe-separated string: ignoreCase|multiline|dotAll. Regex cells are limited to 16,384 characters, patterns to 512, replacement to 4,096 and the complete regex operation has a five-second budget under a disposable supervised worker. Cancellation reaps that worker.")
            FormulaTemporalInformationView()
            Label("Draft sample and captured output",systemImage:"tablecells").fontWeight(.semibold)
            Text("Test on sample executes this draft on at most eight pinned captured upstream rows. Run the upstream flow first. It does not apply changes, read the original source, create a History run or prove all rows will succeed. Truncated display cells are refused. Any draft edit clears that test. Apply commits one undoable edit; unapplied changes block Run. The bottom Results table belongs to the applied run and remains separate from a draft test.")
        }
    }
}
