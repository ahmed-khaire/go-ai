package gemini

import internalhttp "github.com/digitallysavvy/go-ai/pkg/internal/http"

// Config parameterizes the shared Gemini language model implementation
// for both the google and googlevertex providers.
//
// The two providers share identical wire formats, request building,
// response parsing, and streaming logic. Only authentication, base URL,
// and a handful of runtime strings differ — those are captured here and
// injected at construction time, matching the pattern used by the TS SDK
// where GoogleGenerativeAILanguageModel is shared between both packages.
type Config struct {
	// ProviderName is returned by LanguageModel.Provider() and used in error
	// wrapping. "google" for Google Generative AI; "google-vertex" for Vertex AI.
	ProviderName string

	// MetadataKey is the top-level key used in ProviderMetadata output maps.
	// "google" for Google Generative AI; "vertex" for Vertex AI.
	MetadataKey string

	// ProviderOptionsKeys is the ordered list of keys checked when reading
	// caller-supplied provider options from GenerateOptions.ProviderOptions.
	// Google uses ["google"]; Vertex uses ["vertex", "googleVertex", "google"].
	ProviderOptionsKeys []string

	// GeneratePath returns the full HTTP path for a non-streaming request.
	GeneratePath func(modelID string) string

	// StreamPath returns the full HTTP path for a streaming request.
	StreamPath func(modelID string) string

	// Client is the pre-configured HTTP client with auth headers already set.
	Client *internalhttp.Client

	// SupportsCodeExecution enables handling of executableCode and
	// codeExecutionResult parts. True for Google Generative AI, false for Vertex.
	SupportsCodeExecution bool

	// SupportsImageInput returns whether a given model ID accepts image inputs.
	// When nil, the method returns false.
	SupportsImageInput func(modelID string) bool
}
