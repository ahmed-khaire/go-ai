package xai

import "github.com/digitallysavvy/go-ai/pkg/provider/types"

// ViewImage creates a provider-executed tool that analyzes and describes an image.
// Processing is handled by xAI's servers.
//
// Example:
//
//	tool := xai.ViewImage()
func ViewImage() types.Tool {
	return types.Tool{
		Name:             "xai.view_image",
		Description:      "Analyze and describe an image.",
		ProviderExecuted: true,
		Execute:          providerExecutedNoop("xai.view_image"),
	}
}
